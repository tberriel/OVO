from __future__ import annotations
from typing import Any, Dict
import torch
import torch.multiprocessing as mp 
from pathlib import Path
import numpy as np
import time
import os
import gc

from .logger import Logger
from .ovo import OVO
from .datasets import get_dataset
from .visualizer import stream_pcd
from ..slam.vanilla_mapper import VanillaMapper
from ..utils import io_utils

def get_slam_backbone(config: Dict[str, Any], dataset, cam_intrinsics: torch.Tensor) -> VanillaMapper | WrapperGaussianSLAM:
    backbone = config["slam"].get("slam_module","vanilla")
    if backbone == "gaussian_slam":
        from ..slam.gaussian_slam import WrapperGaussianSLAM
        return WrapperGaussianSLAM(config, dataset)
    elif backbone ==  "orbslam2":
        from ..slam.orbslam2 import WrapperORBSLAM2
        return WrapperORBSLAM2(config, cam_intrinsics, world_ref=torch.from_numpy(dataset[0][3]))
    else:
        return VanillaMapper(config, cam_intrinsics)

class OVOSemMap():
    """OVOSemMap is a class responsible for managing the semantic mapping process using the OVO framework. It initializes
    the necessary components, sets up the output path, and handles the main program flow including tracking, mapping,
    and semantic segmentation.
    Args:
        config (dict): Configuration dictionary containing experiment settings.
    """

    def __init__(self, config: Dict[str, Any], output_path: str) -> None:
        self._setup_output_path(output_path)
        io_utils.save_dict_to_yaml(config, "config.yaml", directory=self.output_path)
        config["output_path"] = str(self.output_path)

        self.config = config
        self.device = config.get("device", "cuda")
        self.dataset_name = config["dataset_name"]
        self.stream = self.config["vis"]["stream"]
        self.show_stream = self.config["vis"]["show_stream"]
        self.map_every = config["mapping"].get("map_every", 10)
        self.segment_every = config["semantic"].get("segment_every", 10)
        if config.get("tracking", None) is None:
            self.track_every = 1
        else:
            self.track_every = config["tracking"].get("track_every", 1)

        self.logger = Logger(self.output_path,os.getpid(), config["use_wandb"])   
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})     

        cam_intrinsics =  torch.tensor(self.dataset.intrinsics.astype(np.float32), device=self.device)
        config["semantic"]["debug_info"] = self.config.get("debug_info", False)

        self.ovo = OVO(config["semantic"], self.logger, config["data"]["scene_name"], cam_intrinsics, device=self.device)

        if config["semantic"]["sam"].get("precomputed", False) or config["semantic"]["sam"].get("precompute", False):
            self.ovo.mask_generator.precompute(self.dataset, self.segment_every)
        self.slam_backbone = get_slam_backbone(config, self.dataset, cam_intrinsics)


        self.first_frame = 0
        if self.config.get("restore_map", False):
            assert config["slam"].get("slam_module","vanilla") == "vanilla", "Restoring representation only implemented for 'vanilla' configuration!"
            self.restore_representation()
            self.first_frame = list(self.slam_backbone.estimated_c2ws.keys())[-1]+1

    def _setup_output_path(self, output_path: str) -> None:
        """ Sets up the output path for saving results based on the provided configuration. 
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def save_representation(self) -> None:
        """ Saves the current map and scene objects parameters to a checkpoint file.
        This method retrieves the map parameters from the SLAM backbone and the scene
        objects parameters from the OVO module. It then creates a dictionary containing
        these parameters and saves it to a checkpoint file named after the submap ID.
        The checkpoint file is saved in the 'submaps' directory within the specified
        output path.
        """
        map_params = self.slam_backbone.get_map_dict()
        ovo_map_params = self.ovo.capture_dict(debug_info=self.config.get("debug", False))
        submap_ckpt = {
            "map_params": map_params,
            "ovo_map_params" : ovo_map_params,
        }
        io_utils.save_dict_to_ckpt(
            submap_ckpt, "ovo_map.ckpt", directory=self.output_path)    
        if self.config["slam"].get("save_estimated_cam", False):
            c2w = self.slam_backbone.get_cam_dict()
            with open(self.output_path / "estimated_c2w.npy", "wb") as f:
                torch.save(c2w, f)

    def restore_representation(self) -> None:
        ckpt_path = self.output_path / "ovo_map.ckpt"
        assert ckpt_path.exists(), f"Missing required checkpoint to restore: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.ovo.restore_dict(ckpt["ovo_map_params"], debug_info=self.config.get("debug", False))
        self.slam_backbone.set_map_dict(ckpt["map_params"])
        
        c2w_path = self.output_path / "estimated_c2w.npy"
        if c2w_path.exists():
            c2w = torch.load(c2w_path)
            self.slam_backbone.set_cam_dict(c2w)
        else:
            print(f"Missing cameras positions to restore: {c2w_path}")
            print("Resotring without cameras positions!")
    


    def run(self) -> None:
        """ Starts the main program flow, including tracking and mapping. If self.config["vis"]["stream"] is True, a Open3D visualizer is launched in a parallel process.
        """

        stream = self.config["vis"]["stream"]
        show_stream = self.config["vis"]["show_stream"]
        spf = []

        with mp.Manager() as manager: 
            if stream:
                cam_data = {"height":self.dataset.height, "width": self.dataset.width, "intrinsic": self.dataset.intrinsics} 
                mpqueue = mp.Queue()
                query_flag = mp.Value('i',0) #0 idle, 1 requested, 2 completed
                query_pipe, vis_pipe = mp.Pipe()
                p = mp.Process(target=stream_pcd, args=(self.ovo,mpqueue, [query_flag, vis_pipe],cam_data, self.config["data"]["scene_name"],self.logger.output_path, show_stream), name="O3DVisualizer")
                p.start()

            torch.cuda.synchronize()
            t_start = time.time()
            for frame_id in range(self.first_frame, len(self.dataset)):
                if self.track_every == 1 or frame_id%self.track_every==0 or frame_id%self.map_every==0 or frame_id%self.segment_every==0:
                    frame_data = self.dataset[frame_id]
                    self.slam_backbone.track_camera(frame_data)

                    estimated_c2w = self.slam_backbone.get_c2w(frame_id)
                    missing_depth = not (frame_data[2]>0).any()
                    if estimated_c2w is None or missing_depth :
                        continue
                    t_lc = 0
                    if frame_id % self.map_every == 0 or self.config["slam"]["slam_module"] == "orbslam2":
                        self.slam_backbone.map(frame_data, estimated_c2w)
                        if self.slam_backbone.map_updated:
                            torch.cuda.synchronize()
                            t_lc_i = time.time()
                            map_data = self.slam_backbone.get_map()
                            kfs = self.slam_backbone.get_kfs()
                            updated_points_ins_ids = self.ovo.update_map(map_data, kfs)
                            if updated_points_ins_ids is not None:
                               self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)
                            self.slam_backbone.map_updated = False
                            torch.cuda.synchronize()
                            t_lc = time.time() - t_lc_i
                            print(f"Sem LC update took {t_lc};")
                    t_sem = 0
                    if frame_id % self.segment_every == 0:
                        t_sem_i = time.time()
                        with torch.inference_mode() and torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                            if len(frame_data)==5:
                                image = frame_data[-1]
                            else:
                                image = frame_data[1]

                            if self.dataset.height != image.shape[0] or  self.dataset.width != image.shape[1]:
                                rgb_depth_ratio = (image.shape[0] / self.dataset.dataset_config["H"], image.shape[1]/self.dataset.dataset_config["W"], self.dataset.crop_edge)
                            else:
                                rgb_depth_ratio = ()

                            scene_data = [frame_id, image, frame_data[2], rgb_depth_ratio]

                            map_data = self.slam_backbone.get_map()
                            updated_points_ins_ids = self.ovo.detect_and_track_objects(scene_data, map_data, estimated_c2w)
                            
                            if updated_points_ins_ids is not None:
                                self.slam_backbone.update_pcd_obj_ids(updated_points_ins_ids)

                            self.ovo.compute_semantic_info()
                            self.logger.log_memory_usage(frame_id)

                        t_sem = time.time()-t_sem_i
                        
                        if stream:
                            pcd, _, pcd_obj_ids = self.slam_backbone.get_map()
                            c2w = self.slam_backbone.get_c2w(frame_id)
                            if c2w is None:
                                continue
                            c2w = c2w.cpu().numpy().astype(np.float16)
                            colors = self.slam_backbone.get_pcd_colors()

                            mpqueue.put([pcd.cpu().numpy().astype(np.float16), pcd_obj_ids.cpu().numpy().astype(np.int16), colors, c2w])

                            if query_flag.value == 1:
                                query = query_pipe.recv()
                                self.ovo.complete_semantic_info()
                                query_map = self.ovo.query(query).cpu().numpy()
                                query_map[query_map<0] = 0
                                with query_flag.get_lock():
                                    query_pipe.send(query_map)
                                    query_flag.value = 2
                    if t_sem+t_lc > 0:
                        spf.append(t_sem + t_lc)         

                    if frame_id % 50 == 0:
                        gc.collect()
                
            self.ovo.complete_semantic_info()
            
            torch.cuda.synchronize()
            t_end = time.time()
            fps = len(self.dataset)/self.segment_every/(t_end-t_start)
            if stream and p.is_alive():
                while mpqueue.qsize()>0 and p.is_alive():
                    if query_flag.value == 1:
                        query = query_pipe.recv()
                        query_map = self.ovo.query(query).cpu().numpy()
                        with query_flag.get_lock():
                            query_pipe.send(query_map)
                            query_flag.value = 2
                    time.sleep(2)
                time.sleep(5)
                
        self.logger.log_fps(fps)
        self.logger.log_spf(spf)
        self.logger.log_max_memory_usage()
        self.logger.write_stats()
        self.logger.print_final_stats()

        self.save_representation()

        self.ovo.cpu()
        del self.slam_backbone, self.ovo
        torch.cuda.empty_cache()

        if self.config["vis"].get("stream", False):
            p.terminate()
            
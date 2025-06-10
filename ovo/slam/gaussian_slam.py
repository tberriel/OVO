from typing import Any, Dict, List, Tuple
from argparse import ArgumentParser
import numpy as np
import torch
import pprint
import os

from .sem_gaussian_model import SemGaussianModel
from ..submodules.gaussian_slam.entities.mapper import Mapper
from ..submodules.gaussian_slam.entities.tracker import Tracker
from ..submodules.gaussian_slam.entities.arguments import OptimizationParams
from ..submodules.gaussian_slam.entities.logger import Logger
class WrapperGaussianSLAM():
    def __init__(self, config: dict, dataset) -> None:
        self.device = config.get("device", "cuda")
        assert self.device == "cuda", "Gaussian SLAM does not support CPU only execution!"
        self.config = config
        self.map_updated = False
        self.close_loops = False
        
        self.dataset = dataset
        self.estimated_c2ws = torch.zeros((len(self.dataset), 4, 4), device=self.device, dtype=torch.float32)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3]).to(self.device)

        self.keyframes_info = {}

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        # Logger setup
        os.makedirs(os.path.join(config["output_path"], "mapping_vis"), exist_ok=True)
        os.makedirs(os.path.join(config["output_path"], "tracking_vis"), exist_ok=True)
        logger = Logger(config["output_path"])
        self.mapper = Mapper(config["mapping"], self.dataset, logger)
        self.tracker = Tracker(config["tracking"], self.dataset, logger)

        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.gaussian_model = SemGaussianModel(sh_degree = 0)
        self.gaussian_model.training_setup(self.opt)

        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id = frame_data[0]
        if frame_id in [0, 1]:
            self.estimated_c2ws[frame_id] = torch.from_numpy(frame_data[3].astype(float)).to(self.device)
        else:
            estimated_c2w = self.tracker.track(
                    frame_id, self.gaussian_model,
                    self.estimated_c2ws[np.array([0, frame_id - 2, frame_id - 1])].cpu().numpy())
            self.estimated_c2ws[frame_id] = torch.from_numpy(estimated_c2w.astype(float)).to(self.device)
    
    def map(self, frame_data: List[Any], estimated_c2w: torch.Tensor) -> None:
        frame_id = frame_data[0]
        print("\nMapping frame", frame_id)
        self.gaussian_model.training_setup(self.opt)
        new_submap = frame_id == 0
        opt_dict = self.mapper.map(frame_id, estimated_c2w.cpu().numpy(), self.gaussian_model, new_submap)
        # Keyframes info update
        self.keyframes_info[frame_id] = {
            "keyframe_id": len(self.keyframes_info.keys()),
            "opt_dict": opt_dict
        }

    def get_c2w(self, frame_id: int) -> None:
        return self.estimated_c2ws[frame_id]

    def get_map(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gaussian_model.get_xyz().detach(), self.gaussian_model.get_ids(), self.gaussian_model.get_obj_ids()

    def get_map_dict(self) -> Dict[str, Any]:
        return self.gaussian_model.capture_dict()

    def update_pcd_obj_ids(self, pcd_objs_ids: torch.Tensor) -> None:
        self.gaussian_model.set_objs_ids(pcd_objs_ids)

    def get_pcd_colors(self) -> np.ndarray:
        return ((self.gaussian_model.get_features().detach()*0.28209+0.5)*255).clip(0).flatten(0,1).cpu().numpy().astype(np.uint8)
    

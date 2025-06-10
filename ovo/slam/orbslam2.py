from typing import Any, Dict, List
import orbslam2
import torch
import os
from pathlib import Path

from .vanilla_mapper import VanillaMapper


def convert_pose(traj, device):
    pose = torch.cat([
        torch.tensor(traj[-12:], device = device).reshape((3,4)),
        torch.tensor([[0,0,0,1]], device = device)
    ])
    return pose


class WrapperORBSLAM2(VanillaMapper):
    """This class uses ORB-SLAM 2 to estimate camera posses and generates a vanilla point-cloud reconstruction by unprojecting depths"""
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)

        self.close_loops = config["slam"].get("close_loops", False)
        self.last_big_change_id = 0
        self.map_updated = False
        self.world_ref = world_ref.to(self.device)
        self.kfs = {}

        configs_path = Path(config["slam"]["config_path"]) / "orbslam2"
        vocab_path = configs_path  / "vocabulary" / "ORBvoc.txt"
        if (configs_path/ config["dataset_name"].lower()/ f"{config['data']['scene_name']}.yaml").exists():
            orbslam_config_path = configs_path / config["dataset_name"].lower()/ f"{config['data']['scene_name']}.yaml"
        else:
            orbslam_config_path = configs_path / f"{config['dataset_name']}.yaml"

        self.orbslam2 = orbslam2.System(str(vocab_path), str(orbslam_config_path), orbslam2.Sensor.RGBD, config["slam"].get("use_viewer",False), not self.close_loops)
        self.orbslam2.initialize()

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        tframe = frame_id
        self.orbslam2.process_image_rgbd(rgb_image, depth_image, tframe) # This actually blocks untill tracking is completed
        tracking_state = self.orbslam2.get_tracking_state()
        if tracking_state == orbslam2.TrackingState.OK:
            orb_c2w = self.orbslam2.get_last_trajectory_point()
            assert int(orb_c2w[0]) == frame_id, "Retrieved wrong frame pose" # This should never happen
            self.estimated_c2ws[frame_id] = self.world_ref@convert_pose(orb_c2w, device = self.device)
        else:
            print(f"Tracking state: {tracking_state}!")
        return 
    
    def map(self, frame_data, c2w) -> None:
        # check if frame is a KeyFrame
        if self.orbslam2.is_last_frame_kf(): # If tracking and maping are parallelized, this call would have a racing condition with ORB-SLAM2 tracking thread
            frame_id = frame_data[0]
            first_p_idx = self.pcd_ids.shape[0]
            super().map(frame_data, c2w)
            last_p_idx = self.pcd_ids.shape[0]
            self.kfs[frame_id] = {"id": frame_id , "pcd_idxs":(first_p_idx, last_p_idx)} # Assumes pcd is not pruned outside of self._update_map

        # detect loop-closure of GBA
        last_big_change_id = self.orbslam2.get_last_big_change_idx()
        # LC and GBA happen one after the other, we could save some computation detecting only GBA
        if self.close_loops and last_big_change_id != self.last_big_change_id:
            self.last_big_change_id = last_big_change_id
            ## DEBUG:
            print(f" Least seen object before: {torch.unique(self.pcd_obj_ids, return_counts=True)[1].min()}")
            self.update_map()
            ## DEBUG:
            print(f" Least seen object after: {torch.unique(self.pcd_obj_ids, return_counts=True)[1].min()}")

    def update_map(self):
        print("Updating dense map ...")
        # update kfs and pcd poses:
        updated_kfs = self.orbslam2.get_keyframe_points()

        new_kfs = {}
        new_pcd = []
        new_pcd_ids = []
        new_pcd_obj_ids = []
        new_pcd_colors = []
        new_c2w = {}
        n_points = 0
        for updated_kf in updated_kfs: 
            # for each keyframe, retrieve our saved keyframe,
            kf_id = int(updated_kf[0])
            kf = self.kfs.get(kf_id)
            if kf is None:
                # Why would a kf not be in self.kfs? They are only deleted when update_map is called
                # but then they shouldn't be anymore in orb_slam kfs list
                # If a Keyframe is added/tracked after orb_slam starts LC/GBA, is it going to be in the retrieved list of kfs?
                continue 

            kf_c2w = self.estimated_c2ws[kf["id"]]
            updated_kf_c2w = self.world_ref@convert_pose(updated_kf[1:13], device = self.device)

            transform = updated_kf_c2w@torch.linalg.inv(kf_c2w)
            # If transform is the identityt matrix then the KF was not modified and this could be skipped.
            # ovo.update_map() could just go over updated KFs' 3D instances to check if they should be fused with other instances
            # Measure mIoU and number of fused instances to evaluate reduced approach.
            updated_kf_pcd = torch.einsum('mn,bn->bm', transform, torch.cat([self.pcd[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]], torch.ones((kf["pcd_idxs"][1]-kf["pcd_idxs"][0],1), device=self.device)], dim=1))[:,:3]

            # kfs that are not in updated_kfs were pruned by ORB_SLAM. They will be removed together with their associated pcd
            old_n_points = n_points
            n_points += len(updated_kf_pcd)
            new_kfs[kf_id] = {"id": kf['id'] , "pcd_idxs":(old_n_points, n_points)}
            new_pcd.append(updated_kf_pcd)
            new_pcd_ids.append(self.pcd_ids[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_pcd_obj_ids.append(self.pcd_obj_ids[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_pcd_colors.append(self.pcd_colors[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_c2w[kf["id"]] = updated_kf_c2w

        self.estimated_c2ws = new_c2w
        self.kfs = new_kfs
        self.pcd = torch.cat(new_pcd, dim=0)
        self.pcd_ids = torch.cat(new_pcd_ids, dim=0)
        self.pcd_obj_ids = torch.cat(new_pcd_obj_ids, dim=0)
        self.pcd_colors = torch.cat(new_pcd_colors, dim=0)
        self.map_updated = True

    

    def __del__(self) -> None:
        self.orbslam2.shutdown()
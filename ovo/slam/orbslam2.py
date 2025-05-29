from typing import Any, Dict, List
import orbslam2
import torch
import os

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

        self.close_loops = config.get("close_loops", False)
        self.last_big_change_id = 0

        self.world_ref = world_ref.to(self.device)
        self.kfs = {}

        vocab_path = os.path.join(config["slam"]["config_path"], "orbslam2", "vocabulary/ORBvoc.txt")
        config_path = os.path.join(config["slam"]["config_path"], "orbslam2", config["dataset_name"]+".yaml")
        self.orbslam2 = orbslam2.System(vocab_path, config_path, orbslam2.Sensor.RGBD)
        self.orbslam2.set_use_viewer(config["slam"].get("use_viewer",False))
        self.orbslam2.initialize()

    
    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        tframe = frame_id
        self.orbslam2.process_image_rgbd(rgb_image, depth_image, tframe)
        tracking_state = self.orbslam2.get_tracking_state()
        if tracking_state == orbslam2.TrackingState.OK:
            orb_c2w = self.orbslam2.get_trajectory_points()[-1]# TODO: Make function to return only frame_id frame pose
            assert int(orb_c2w[0]) == frame_id, "Retrieved wrong frame pose"
            self.estimated_c2ws[frame_id] = self.world_ref@convert_pose(orb_c2w, device = self.device)
            # check if frame is a KeyFrame
            if self.orbslam2.is_last_frame_kf():
                # map
                self._map(frame_data)
                # detect loop-closure of GBA
                last_big_change_id = self.orbslam2.get_last_big_change_idx()
                if self.close_loops and last_big_change_id != self.last_big_change_id:
                    self.last_big_change_id = last_big_change_id
                    self._update_map()
        else:
            print(f"Tracking state: {tracking_state}!")
        return 
    
    def map(self, *args, **kwargs) -> None:
        pass # map is launched by track_camera calling _map

    def _map(self, frame_data):
        frame_id = frame_data[0]
        first_p_idx = self.pcd_ids.shape[0]
        super().map(frame_data, self.estimated_c2ws[frame_id])
        last_p_idx = self.pcd_ids.shape[0]
        self.kfs[frame_id] = {"id": frame_id , "pcd_idxs":(first_p_idx, last_p_idx)} # Assumes pcd is not pruned outside of self._update_map

    def _update_map(self):
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

            kf_c2w = self.estimated_c2ws[kf["id"]]
            updated_kf_c2w = self.world_ref@convert_pose(updated_kf[1:13], device = self.device)

            transform = updated_kf_c2w@torch.linalg.inv(kf_c2w)
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
        
    
    def __del__(self) -> None:
        self.orbslam2.shutdown()
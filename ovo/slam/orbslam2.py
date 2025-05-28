from typing import Any, Dict, List
import orbslam2
import torch
import os

from .vanilla_mapper import VanillaMapper


def convert_pose(traj, device):
    _, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
    pose = torch.tensor([[r00, r01, r02, t0],
                        [r10, r11, r12, t1],
                        [r20, r21, r22, t2],
                        [0, 0, 0, 1]], device = device)
    return pose


class WrapperORBSLAM2(VanillaMapper):
    """This class uses ORB-SLAM 2 to estimate camera posses and generates a vanilla point-cloud reconstruction by unprojecting depths"""
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)

        self.world_ref = world_ref.to(self.device)

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
            orb_c2w = self.orbslam2.get_trajectory_points()[-1]
            self.estimated_c2ws[frame_id] = self.world_ref@convert_pose(orb_c2w, device = self.device)
        else:
            print(f"Tracking state: {tracking_state}!")
        return 
    
    def __del__(self) -> None:
        self.orbslam2.shutdown()
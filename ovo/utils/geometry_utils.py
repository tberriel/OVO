from torchvision.transforms.v2.functional import gaussian_blur
from typing import Any, Dict, List, Tuple
import numpy as np
import torch


def match_3d_points(cluster_0: torch.Tensor, cluster_1: torch.Tensor, dist_th: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Match each point in cluster_0 to its closest point in cluster_1. Only matches if dist < dist_th
    Args: 
        - cluster_0 (mx3): tensor with m coordiantes of 3 d points
        - cluster_1 (nx3): tensor with n coordiantes of 3 d points
    Return:
        - mask (bool): mask of i<m points in cluster_0 with a match
        - matches (ix1): tensor of idexes in cluster_1 of each match from cluster_0.
    """

    n = cluster_1.shape[0]
    #compute distance
    dist = torch.norm(cluster_0.unsqueeze(1).repeat(1,n,1).cpu() - cluster_1.cpu(), dim=-1) # norm(mxnx3 - nx3) -> mxn
    
    #filter points further away than th_dist to its closest match
    mask = ((dist<dist_th).sum(-1)>0) # mxn -> m, from which i<m have a match
    matches = dist[mask,...].argmin(-1) # ix1
    return mask, matches

def project_3d_points(points_3d: torch.Tensor, intrinsics: torch.Tensor, w2c: torch.Tensor = None) -> torch.Tensor :
    """ Project 3D points to the 2D camera frame. Move points to camera reference if world to camera (w2c) transform is passed, else assume points are already in camera reference.
    Args:
        points_3d (torch.Tensor) : Nx4 list of 3D points in homogeneous coordinates
        intrinsics (torch.Tensor) : 3 x 3 camera's intrinsic projection matrix
        w2c (optional) (torch.Tensor) : 4 x 4 transform from points reference to camera reference
    Returns:
        points_2d (torch.Tensor) : Nx2 list of 2D positions of 3D ponts in the camera frame, as inhomogeneous integer coordinates
    """

    if w2c is not None:
        points_3d = torch.einsum("mn,bn->bm", w2c, points_3d)
    points_3d = points_3d[...,:3]/points_3d[...,3:]


    points_2d = torch.einsum("mn,bn->bm", intrinsics, points_3d)

    return (points_2d[:,:2]/points_2d[:,2:]).round().int()

 
def match_3d_points_to_2d_pixels(depth: torch.Tensor, w2c: torch.Tensor, points_3d: torch.Tensor, intrinsics: torch.Tensor, th_dist: float) -> tuple[torch.Tensor, torch.Tensor]: 
    """ Match 3D points to 2D camera coordinates in pixels. First move gaussian to local reference, filter out non-forward gaussians, project gaussians to 2d plane, match gaussians with projected 2d coordinates if depth distance < th_dist.
    Args:
        - depth (torch.Tensor): image depth.
        - w2c (torch.Tensor): 4 x 4 transform from points reference to camera reference.
        - points_3d (torch.Tensor): Nx4 list of 3D points in homogeneous coordinates.
        - intrinsic (torch.Tensor): 3 x 3 camera's intrinsic projection matrix.
        - th_dist (float): maximum distance between 3d point and depth sensor.
    Returns:
        mask (torch.tensor): idxs in points_3d of N matched points.
        matches (torch.tensor): (N,2) tensor containing for each 3D point its 2D match.
    """
    h,w = depth.shape
    device = points_3d.device
    n_points = points_3d.shape[0]
    idx = torch.tensor(list(range(n_points)), device=device)
    # move 3d points to local reference
    if points_3d.shape[-1] == 3:
        points_3d = torch.hstack([points_3d,torch.ones((n_points,1), device=device)])

    local_points_3d = torch.einsum("mn,bn->bm", w2c, points_3d)
    
    # Mask 3d points not in front of the camera
    forward_points_3d = local_points_3d # Points inside frustum already filtered
    
    # Project 3d points to the camera frame
    points_2d = project_3d_points(forward_points_3d, intrinsics)
    in_points_2d = points_2d # Points inside frustum already filtered
    # Mask 2d points out of camera frame
    in_plane_mask = torch.logical_and(points_2d[:,0]< w, points_2d[:,1]<h)*torch.logical_and(points_2d[:,0]>=0, points_2d[:,1]>=0) # Points inside frustum already filtered
    in_points_2d = points_2d[in_plane_mask] # Points inside frustum already filtered

    # Compute depth of 3d points for points projected inside camera frame
    forward_points_depth = forward_points_3d[in_plane_mask,2] # depth as z coordinate # Points inside frustum already filtered

    # Mask 3d points with distance between their depth and gt depth at the projected point bigger than a threshold 
    dist_mask = (forward_points_depth - depth[in_points_2d[:,1],in_points_2d[:,0]]).abs() < th_dist
    dist_mask[depth[in_points_2d[:,1],in_points_2d[:,0]] == 0] = False # We could skip this step because invalid depth values shouldbe filterd by the distance threshold.
    # For 3d points inside the distance threshold, asing the projected 2d point as a match
    matches = in_points_2d[dist_mask]        

    # propagate final mask to total 3d points list. 
    mask = idx[in_plane_mask][dist_mask]
    return mask, matches


def depth_filter(depth: torch.Tensor, k_size: int = 7,sigma: float= 2.5, th: float = 0.05) -> torch.Tensor:
    low_frequencies = gaussian_blur(depth[None], k_size, sigma)[0]
    high_frequencies = (depth-low_frequencies).abs()
    masked_depth = torch.where(high_frequencies>th, -1, depth)
    return masked_depth


def compute_camera_frustum_corners(depth_map: torch.Tensor, pose: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """ Code from Gaussian-SLAM mapper_utils 
     Computes the 3D coordinates of the camera frustum corners based on the depth map, pose, and intrinsics.
    Args:
        depth_map: The depth map of the scene.
        pose: The camera pose matrix.
        intrinsics: The camera intrinsic matrix.
    Returns:
        An array of 3D coordinates for the frustum corners.
    """
    height, width = depth_map.shape
    depth_map = depth_map[depth_map > 0]
    min_depth, max_depth = depth_map.min(), depth_map.max()
    corners = torch.tensor(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ], device=depth_map.device
    )
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]
    corners_3d = torch.vstack((x, y, z, torch.ones(x.shape[0], device=depth_map.device))).T
    corners_3d = torch.einsum("ij,mj->mi", pose, corners_3d)
    return corners_3d[:, :3]

def compute_camera_frustum_corners_cpu(depth_map: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ Code from Gaussian-SLAM mapper_utils 
     Computes the 3D coordinates of the camera frustum corners based on the depth map, pose, and intrinsics.
    Args:
        depth_map: The depth map of the scene.
        pose: The camera pose matrix.
        intrinsics: The camera intrinsic matrix.
    Returns:
        An array of 3D coordinates for the frustum corners.
    """
    height, width = depth_map.shape
    depth_map = depth_map[depth_map > 0]
    min_depth, max_depth = depth_map.min(), depth_map.max()
    corners = np.array(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ]
    )
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]
    corners_3d = np.vstack((x, y, z, np.ones(x.shape[0]))).T
    corners_3d = pose @ corners_3d.T
    return corners_3d.T[:, :3]

def compute_camera_frustum_planes(frustum_corners: np.ndarray) -> torch.Tensor:
    """ Code from Gaussian-SLAM mapper_utils 
     Computes the planes of the camera frustum from its corners.
    Args:
        frustum_corners: An array of 3D coordinates representing the corners of the frustum.

    Returns:
        A tensor of frustum planes.
    """
    # near, far, left, right, top, bottom
    planes = torch.stack(
        [
            torch.linalg.cross(
                frustum_corners[2] - frustum_corners[0],
                frustum_corners[1] - frustum_corners[0],
            ),
            torch.linalg.cross(
                frustum_corners[6] - frustum_corners[4],
                frustum_corners[5] - frustum_corners[4],
            ),
            torch.linalg.cross(
                frustum_corners[4] - frustum_corners[0],
                frustum_corners[2] - frustum_corners[0],
            ),
            torch.linalg.cross(
                frustum_corners[7] - frustum_corners[3],
                frustum_corners[1] - frustum_corners[3],
            ),
            torch.linalg.cross(
                frustum_corners[5] - frustum_corners[1],
                frustum_corners[3] - frustum_corners[1],
            ),
            torch.linalg.cross(
                frustum_corners[6] - frustum_corners[2],
                frustum_corners[0] - frustum_corners[2],
            ),
        ]
    )
    D = torch.stack([-torch.dot(plane, frustum_corners[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, D[:, None]], dim=1).float()


def compute_frustum_aabb(frustum_corners: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Code from Gaussian-SLAM mapper_utils 
    Computes a mask indicating which points lie inside a given axis-aligned bounding box (AABB).
    Args:
        points: An array of 3D points.
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        Frustum corners
    """
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values


def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """ Code from Gaussian-SLAM mapper_utils 
    Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2]))



def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    """ Code from Gaussian-SLAM mapper_utils 
    Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    num_pts = points.shape[0]
    ones = torch.ones(num_pts, 1).to(points.device)
    plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
    return torch.all(plane_product <= 0, axis=1)


def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """ Code from Gaussian-SLAM mapper_utils 
    Identifies points within the camera frustum, optimizing for computation on a specified device.
    Args:
        pts: A tensor of 3D points.
        frustum_corners: A tensor of 3D coordinates representing the corners of the frustum.
        device: The computation device ("cuda" or "cpu").
    Returns:
        Indices of points lying inside the frustum.
    """
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    # Broad phase
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner)

    # Narrow phase
    frustum_planes = compute_camera_frustum_planes(frustum_corners)
    frustum_planes = frustum_planes.to(device)
    inside_frustum_mask = points_inside_frustum_mask(pts[inside_aabb_mask], frustum_planes)

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask
    return torch.where(inside_aabb_mask)[0]
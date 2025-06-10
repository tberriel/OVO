import open3d as o3d
import torch
import numpy as np

def same_instance(instance1, instance2, map_data, th_centroid = 0.05, th_cossim = 0.9, th_points = 0.05):
    points_3d, points_ids, points_ins_ids = map_data
    # Check centroids
    points1, points2 = points_3d[points_ins_ids == instance1.id], points_3d[points_ins_ids == instance2.id]
    centroid1 = (points1).mean(axis=0)
    centroid2 = (points2).mean(axis=0)
    distance = ((centroid1 - centroid2) ** 2).sum().sqrt()
    if distance > th_centroid:
        return False
    # Check CLIP similarity
    if torch.nn.functional.cosine_similarity(instance1.clip_feature[0], instance2.clip_feature[0], dim=0) < th_cossim:
        return
    # Check if more than 50% of points are really close
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))

    return (dists < th_points).astype(float).mean() > 0.5

def fuse_instances(instance1, instance2, map_data):
    points_3d, points_ids, points_ins_ids = map_data
    
    instance1.add_points_ids(instance2.points_ids)
    for kf in instance2.kfs_ids:
        instance1.add_keyframes(kf)
    for (area, kf_id) in instance2.top_kf:
        instance1.add_top_kf(kf_id, area)
    points_ins_ids[points_ins_ids == instance2.id] = instance1.id
    return instance1, points_ins_ids
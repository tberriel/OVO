from pathlib import Path
import numpy as np
import argparse
import torch
import yaml

from ovo.entities.visualizer import visualize_3d_points_obj_id_and_obb, visualize_gt_vs_pred, Visualizer   
from ovo.utils.io_utils import load_config, load_scene_data, read_labels
from .run_eval import load_representation

def capitalize_first(text):
  return text[0].upper() + text[1:]

def main(args):
    run_path = Path(args.working_dir) / args.run_path
    config = load_config(run_path/"config.yaml")

    semantic_module, params, pcd_pred = load_representation(run_path, eval=True)
    obj_ids = params["obj_ids"]

    dataset_name = capitalize_first(config["dataset_name"])
    if dataset_name == "Scannet":
        dataset_name = "ScanNet"
    path = Path(args.working_dir) / "data/working/configs/" / dataset_name  / args.dataset_info_file
    data_path = Path(args.working_dir) / "data/input/Datasets/"
    with open(path, 'r') as f:
        dataset_info = yaml.full_load(f)
    
    classes = dataset_info["class_names_reduced"] if dataset_info.get("map_to_reduced", None)  else dataset_info["class_names"] 
    pcd_labels_gt, pcd_gt = load_scene_data(config["dataset_name"], config["data"]["scene_name"], data_path, dataset_info)
    pcd_labels_pred = read_labels(run_path.parent / dataset_info["dataset"] / (config["data"]["scene_name"]+".txt"))

    map_to_reduced = dataset_info.get("map_to_reduced", None)
    scene_labels_idxs = np.unique(pcd_labels_gt)

    if map_to_reduced is not None and dataset_name != "Replica":
        for id in scene_labels_idxs:
            if id not in map_to_reduced.keys():
                map_to_reduced[id] = -1
        pcd_labels_gt = np.vectorize(map_to_reduced.get)(pcd_labels_gt)
        scene_labels_idxs = np.unique(pcd_labels_gt)
    pcd_labels_gt = np.asarray(pcd_labels_gt)

    while scene_labels_idxs[0]<0:
        scene_labels_idxs = scene_labels_idxs[1:]
    classes = np.array(classes)[scene_labels_idxs]
    
    sh_c0 = 0.28209479177387814
    if args.visualize_obj or args.visualize_interactive_query or args.visualize_gt_vs_pre:
        if params.get("features_dc", None) is not None:
            pcd_colors = (params["features_dc"]*sh_c0+0.5).clip(0).flatten(0,1)
        elif params.get("color") is not None:
            pcd_colors = params["color"]
        else:
            pcd_colors = torch.rand(pcd_pred.shape)*255
            
        while True:
            if args.visualize_obj:
                visualize_3d_points_obj_id_and_obb(pcd_pred, obj_ids, pcd_colors)
            if args.visualize_interactive_query:
                vis = Visualizer(semantic_module, scene_name=config["data"]["scene_name"], save_path=run_path.parent)
                vis.visualize_and_query(pcd_pred, params["obj_ids"].squeeze().numpy(), pcd_colors)
            if args.visualize_gt_vs_pre:
                mask = pcd_labels_gt>=0
                visualize_gt_vs_pred(pcd_gt[mask], pcd_labels_gt[mask], pcd_labels_pred[mask].astype(np.int64), np.array(classes), scene_labels_idxs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to train and evaluate over a dataset')
    parser.add_argument('run_path', type=str)
    parser.add_argument('--working_dir', default="/home/tberriel/Workspaces/semsplat_ws/sem3d", type=str)
    parser.add_argument('--visualize_obj', action='store_true')
    parser.add_argument('--visualize_interactive_query', action='store_true')
    parser.add_argument('--visualize_gt_vs_pre', action='store_true')
    parser.add_argument('--dataset_info_file',type=str, default="eval_info.yaml")
    args = parser.parse_args()
    main(args)
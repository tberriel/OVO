import os
from pathlib import Path
from tqdm import tqdm
import plyfile
import argparse
import numpy as np


def write_labels(output_file: str, pcd_labels: np.ndarray) -> None:
    n_vtx = pcd_labels.shape[0]
    labels_list = [str(int(pcd_labels[i].item())) for i in range(n_vtx)]
    with open(output_file, "w") as f:
        f.write('\n'.join(labels_list))

def main(args):
    scannet_data_path = Path(args.data_path)
    scans_path = scannet_data_path / "scans"

    decoded_path = scannet_data_path / "data"/ "val"
    scannet200_path = scannet_data_path / "scannet200" / "val"

    scenes = os.listdir(decoded_path)
    scenes = [scene for scene in scenes if scene[:5] == "scene"]

    if args.scannet200:
        out_path = decoded_path / "scannet200_gt"
    else:
        out_path = decoded_path / "semantic_gt"

    os.makedirs(out_path, exist_ok=True)

    for scene in tqdm(scenes):
        if args.scannet200:
            mesh_path = scannet200_path / f"{scene}.ply"
        else:
            mesh_path = scans_path / scene  / f"{scene}_vh_clean_2.labels.ply"
            if args.link_pcds:
                os.symlink(mesh_path, decoded_path / scene / f"{scene}_vh_clean_2.labels.ply")

        plydata = plyfile.PlyData.read(mesh_path)
        gt_labels = np.array(plydata["vertex"]["label"])
        write_labels(out_path / f"{scene}.txt", gt_labels)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to directory containing scans and data folder')
    parser.add_argument('--link_pcds', action='store_true' , help='If set, creates a symbolic link to gt pointclouds in data/val/scene*/. If --scannet200 is set, this is ignored.')
    parser.add_argument('--scannet200', action='store_true' , help='If set, only saves ScanNet200 labels.')
    args = parser.parse_args()
    main(args)
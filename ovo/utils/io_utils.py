from typing import Dict, Any, Tuple, Union
from pathlib import Path
import open3d as o3d
import numpy as np
import plyfile
import torch
import math
import json
import yaml
import os


def load_config(path: str, default_path: str = None, inherit: bool = True) -> Dict[str, Any]:
    """
    Loads a configuration file and optionally merges it with a default configuration file.

    This function loads a configuration from the given path. If the configuration specifies an inheritance
    path (`inherit_from`), or if a `default_path` is provided, it loads the base configuration and updates it
    with the specific configuration.

    Args:
        path: The path to the specific configuration file.
        default_path: An optional path to a default configuration file that is loaded if the specific configuration
                      does not specify an inheritance or as a base for the inheritance.

    Returns:
        A dictionary containing the merged configuration.
    """
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    inherit_from = cfg_special.get('inherit_from')
    cfg = dict()
    if inherit_from is not None and inherit:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    update_recursive(cfg, cfg_special)
    return cfg

def update_recursive(dict1: Dict[str,Any], dict2: Dict[str,Any]) -> None:
    """ Recursively updates the first dictionary with the contents of the second dictionary.

    This function iterates through `dict2` and updates `dict1` with its contents. If a key from `dict2`
    exists in `dict1` and its value is also a dictionary, the function updates the value recursively.
    Otherwise, it overwrites the value in `dict1` with the value from `dict2`.

    Args:
        dict1: The dictionary to be updated.
        dict2: The dictionary whose entries are used to update `dict1`.

    Returns:
        None: The function modifies `dict1` in place.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_scene_data(dataset_name: str, scene_name: str, data_path: str, dataset_info: Dict[str, Any], ignore_background: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if dataset_name == "ScanNetpp" or dataset_name.lower() == "scannetpp":
        return load_scannetpp_scene(scene_name, data_path)
    elif dataset_name == "Replica" or dataset_name.lower() == "replica":
        return load_replica_scene(scene_name, data_path, dataset_info, ignore_background)
    elif dataset_name == "ScanNet" or dataset_name.lower() == "scannet":
        return load_scannet_scene(scene_name, data_path, dataset_info.get("dataset", "scannet"))
    else:
        assert False, f"{dataset_name} dataset not implemented"

def load_scannetpp_scene(scene_name: str, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data_folder = Path(data_path) / "ScanNetpp/" 
    mesh_path = data_folder / "semantic/eval_meshes" / (scene_name+".pth")
    gt_labels_path = data_folder / "semantic/eval_labels" / (scene_name+".txt")

    with open(gt_labels_path, "r") as f:
        gt_labels = f.read().splitlines() 
    gt_labels = torch.tensor([int(i) for i in gt_labels])

    mesh = torch.load(mesh_path) 
    vtx_coords = mesh["vtx_coords"]

    # Move mesh coordinates rotating z axis -90 degrees
    P = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    vtx_coords = torch.einsum("mn,bn->bm",P,torch.cat([vtx_coords, torch.ones(vtx_coords.shape[0],1)],dim=-1))
    vtx_coords = vtx_coords[:,:3]/vtx_coords[:,3:]

    return gt_labels, vtx_coords

def load_scannet_scene(scene_name: str, data_path: str, version: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if version == "scannet200":
        gt_labels = np.array(read_labels( Path(data_path) / "ScanNet"/ "scannet200_gt" / f"{scene_name}.txt"))
    else:
        gt_labels = np.array(read_labels( Path(data_path) / "ScanNet"/ "semantic_gt" / f"{scene_name}.txt"))

    mesh_path = Path(data_path) / "ScanNet"/ scene_name /  f"{scene_name}_vh_clean_2.labels.ply"
    plydata = plyfile.PlyData.read(mesh_path)
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    return gt_labels, vertices

def load_replica_scene(scene_name: str, data_path: str, dataset_info: Dict[str, Any], ignore_background: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    map_to_reduced = dataset_info.get("map_to_reduced", None)
    ignore = dataset_info.get("ignore",[])
    if ignore_background:
        assert dataset_info["background_reduced_ids"], "To ignore background a list of idxs corresponding to background ids id required!"
        ignore.extend(dataset_info["background_reduced_ids"])

    data_folder = Path(data_path) /"Replica/" 

    gt_labels_path = data_folder / "semantic_gt" / f"{scene_name}.txt"
    with open(gt_labels_path, "r") as f:
        gt_labels = f.read().splitlines() 
    gt_labels = np.array([int(i) for i in gt_labels])
    if map_to_reduced is not None:
        gt_labels = torch.tensor(np.vectorize(map_to_reduced.get)(gt_labels))
    for id_to_ignore in ignore:
        gt_labels[gt_labels==id_to_ignore] = -100

    mesh_path = data_folder / f'{scene_name}_mesh.ply'
    scene_point_cloud = o3d.io.read_point_cloud(str(mesh_path))
    pcd = torch.tensor(np.array(scene_point_cloud.points))
    return gt_labels, pcd

def rle_encode(mask: np.ndarray) -> Dict[str, Any]:
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle

def rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def write_instances(experiment_path: str,scene_name: str, instances_info: Dict[str, Any]) -> None:
    save_path = os.path.join(experiment_path, "instance_pred")
    os.makedirs(save_path, exist_ok=True)
    
    rel_path = "./predicted_masks/"
    os.makedirs(os.path.join(save_path, rel_path), exist_ok=True)
    n_instances = len(instances_info["masks"])
    n_digits = math.trunc(math.log(n_instances,10)) +1
    lines_to_write = []
    for i in range(n_instances):
        mask = instances_info["masks"][i]
        label = instances_info["classes"][i]
        conf = instances_info["conf"][i]

        mask_file = os.path.join(rel_path, f"{scene_name}_{str(i).zfill(n_digits)}.json")
        with open(os.path.join(save_path,mask_file), "w") as f:
            rle = rle_encode(mask)
            json.dump(rle, f)

        lines_to_write.append(f"{mask_file} {int(label)} {conf:.4f}")

    with open(os.path.join(save_path, f"{scene_name}.txt"), "w") as f:
            f.write('\n'.join(lines_to_write))

def write_labels(output_file: str, pcd_labels: np.ndarray) -> None:
    n_vtx = pcd_labels.shape[0]
    labels_list = [str(int(pcd_labels[i].item())) for i in range(n_vtx)]
    with open(output_file, "w") as f:
        f.write('\n'.join(labels_list))

def read_labels(output_file: str) -> np.ndarray:
    with open(output_file, "r") as f:
        labels_list = f.read().splitlines()
    labels =  np.array(labels_list).astype(np.int64)
    return labels

def mkdir_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper

@mkdir_decorator
def save_dict_to_ckpt(dictionary: Dict[str, Any], file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a checkpoint file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the checkpoint file.
        directory: The directory where the checkpoint file will be saved.
    """
    try:
        torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)
    except OverflowError as e:
        torch.save(dictionary, directory / file_name,
               pickle_protocol=4)


@mkdir_decorator
def save_dict_to_yaml(dictionary: Dict[str, Any], file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a YAML file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the YAML file.
        directory: The directory where the YAML file will be saved.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)
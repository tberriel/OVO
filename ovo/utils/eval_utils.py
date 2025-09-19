from typing import Any, Dict, List, Tuple
from sklearn.neighbors import BallTree
from matplotlib.colors import LogNorm
from scipy.spatial import KDTree
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn # seaborn is changing matplotlib configuration to automatically open graphs without calling plt.imshow()
import pandas as pd
import numpy as np
import torch 
import sys

def match_labels_to_vtx(points_3d_labels: torch.Tensor, points_3d: torch.Tensor, mesh_vtx: torch.Tensor, filter_unasigned: bool = True, tree: str ="kd", verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # assume points_3d and mesh_vtx in the same reference frame
    if filter_unasigned:
        assigned_mask = (points_3d_labels >-1).squeeze()
        if verbose:
            print(f"Assigned points {assigned_mask.sum()}, {assigned_mask.float().mean()*100:.1f}")
        points_3d_labels = points_3d_labels[assigned_mask]
        points_3d = points_3d[assigned_mask]
        assert len(points_3d_labels), "All points are unassigned"
    
    if tree =="ball":
        tree = BallTree(points_3d)
    else:
        tree = KDTree(points_3d)
    distances, indices = tree.query(mesh_vtx, k=5)

    labels = points_3d_labels[indices]
    mesh_labels = torch.mode(labels).values
    
    # compute instance masks
    # It may happen that not all instances matched to the predicted point cloud, are matched with the GT mesh. Therefore let's use ids rather than idxs
    matched_instances_ids = torch.unique(mesh_labels)
    if not filter_unasigned:
        # Filter unassigned ids
        while matched_instances_ids[0]<0:
            matched_instances_ids = matched_instances_ids[1:]
        
    n_instances = len(matched_instances_ids)
    instance_idxs = torch.unsqueeze(matched_instances_ids, dim=1)
    mesh_instances_masks = torch.unsqueeze(mesh_labels,dim=0).expand(n_instances,-1) == instance_idxs

    return mesh_labels, mesh_instances_masks, matched_instances_ids

def plot_metrics(iou_values: np.ndarray, acc_values: np.ndarray, labels: List, output_path: Path, ignore: List = []) -> None:
    labels = [label for i,label in enumerate(labels) if i not in ignore]
    idx = np.asarray([0.4+i*3 for i in range(len(labels))])
    width = 1.
    ratio =10/len(labels)
    fig, axs = plt.subplots(figsize=(20,40*ratio))
    axs.margins(x=0.01)
    axs.set_title("IoU and Acc")
    axs.set_box_aspect(ratio)  # More width than height
    axs.bar(idx, iou_values, width=width)
    axs.bar(idx+width, acc_values, width=width)
    axs.set_xticks(idx)
    axs.set_xticklabels(labels, rotation=85)
    axs.legend(['IoU', 'Acc'], loc='upper right')

    #plt.show()
    plt.tight_layout()
    plt.savefig(output_path / "plot_iou_acc.png")
    plt.close()

def plot_confmat(confmat: np.ndarray, labels: List, output_path: Path, save: bool = True) -> None:
    n =len(labels)
    fig, axs = plt.subplots(figsize=(0.32*n,0.28*n)) #16,14
    axs.set_title(f"Confusion matrix.")    
    df_cm = pd.DataFrame(confmat, index = labels,columns = labels)
    sn.heatmap(df_cm, annot=False, ax = axs, xticklabels=True, yticklabels=True, fmt=".1g", norm=LogNorm())
    sn.set(font_scale=3)
    plt.tight_layout()
    if save:
        plt.savefig(output_path / "confmat.png")
    else:
        plt.show()

    plt.close()
    return

def process_txt(filename: str) -> List[str]:
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines
    
def evaluate_scan(pr_file: str, gt_file: str, confusion: np.ndarray, map_gt_ids: Dict[int, int] | None = None, ignore: List = []) -> None:

    pr_ids = np.array(process_txt(pr_file), dtype=np.int64)
    gt_ids = np.array(process_txt(gt_file)).astype(np.int64)

    if map_gt_ids is not None:
        assert isinstance(map_gt_ids,dict), "map_gt_ids must be either None or a dictionary which keys map gt label idxs to new idxs"
        ids = np.unique(gt_ids)
        for id in ids:
            if id not in map_gt_ids.keys():
                map_gt_ids[id] = -1
        gt_ids = np.vectorize(map_gt_ids.get)(gt_ids)

    # sanity checks
    if not pr_ids.shape == gt_ids.shape:
        print(f'number of predicted values does not match number of vertices. pred: {pr_ids.shape}; gt: {gt_ids.shape};{pr_file}')
    """
    for (gt_val, pr_val) in zip(gt_ids, pr_ids):
        if gt_val in ignore:
            continue
        confusion[gt_val][pr_val] += 1"""
    update_confmat(confusion, gt_ids, pr_ids, ignore)

def update_confmat(confusion: np.ndarray, gt_ids: List[int], pr_ids: List[int], ignore: List[int]) -> None:
    for (gt_val, pr_val) in zip(gt_ids, pr_ids):
        if gt_val in ignore:
            continue
        confusion[gt_val][pr_val] += 1

def get_iou(label_id: int, confusion: np.ndarray) -> Tuple[float, float]:
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return (float('nan'),float('nan'))
    iou = tp / denom
    acc = tp / max(float(tp + fn), 1e-6)
    return (iou, acc)


def iou_acc_from_confmat(confmat: np.ndarray, num_classes: int, ignore: List[int], mask_nan: bool = True, verbose: bool = False, labels: List[str] = None):

    if verbose:
        print('\n classes \t IoU \t Acc')
        print('----------------------------')
    list_iou, list_acc, list_weight = [], [], []
    for i in range(num_classes):
        if i not in ignore:
            iou, acc = get_iou(i, confmat)
            list_iou.append(iou)
            list_acc.append(acc)
            list_weight.append(confmat[i].sum()) # frequency or tp + fn 
            if verbose:
                label_name = labels[i]
                print('{0:<14s}: {1:>5.2%}   {2:>6.2%}'.format(label_name, iou, acc))  

    iou_values = np.array(list_iou)
    acc_values = np.array(list_acc)
    weights_values = np.array(list_weight)

    if mask_nan:
        iou_valid_mask = ~np.isnan(iou_values)
        acc_valid_mask = ~np.isnan(acc_values)
    else:
        iou_valid_mask = np.ones_like(iou_values,dtype=bool)
        acc_valid_mask = np.ones_like(acc_values,dtype=bool)
    return iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask

def eval_semantics(output_path: str, gt_path: str, scenes: List[str], dataset_info: Dict[str, Any], mask_nan: bool = True, ignore_background: bool = False, verbose: bool = True, return_metrics = False) -> Tuple[np.ndarray, np.ndarray]:
    
    num_classes = dataset_info["num_classes"]
    map_to_reduced = dataset_info.get("map_to_reduced", None)
    labels = dataset_info["class_names"] if dataset_info.get("map_to_reduced", None) is None else dataset_info["class_names_reduced"]
    ignore = dataset_info.get("ignore",[]).copy()
    
    if ignore_background:
        if map_to_reduced:
            assert dataset_info.get("background_reduced_ids", None), "To ignore background a list of idxs corresponding to background ids id required!"
            ignore.extend(dataset_info["background_reduced_ids"])
        else:
            assert dataset_info["background_ids"], "To ignore background a list of idxs corresponding to background ids id required!"
            ignore.extend(dataset_info["background_ids"])
    #valid_labels = [label for i,label in enumerate(labels) if i not in ignore]

    pr_files = []  # predicted files
    gt_files = []  # ground truth files
    for scene in scenes:
        pr_files.append(Path(output_path)/ f'{scene}.txt')
        gt_files.append(Path(gt_path) / f'{scene}.txt')
    
    confusion = np.zeros([len(scenes), num_classes, num_classes], dtype=np.ulonglong)

    if verbose:
        print('evaluating', len(pr_files), 'scans...')
    for i in range(len(pr_files)):
        evaluate_scan(pr_files[i], gt_files[i], confusion[i], map_to_reduced, ignore)
        if verbose:
            sys.stdout.write("\rscans processed: {}".format(i+1))
            sys.stdout.flush()

    # Per scene:
    for i in range(len(scenes)):
        iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = iou_acc_from_confmat(confusion[i], num_classes, ignore, mask_nan, False, labels)
        if verbose:
            print(f"Scene: {scenes[i]}")
            print(f'mIoU: \t {np.mean(iou_values[iou_valid_mask]):.2%}; mAcc: \t {np.mean(acc_values[acc_valid_mask]):.2%}\n ')
            print(f'f-mIoU: \t {np.sum(iou_values[iou_valid_mask]*weights_values[iou_valid_mask])/weights_values[iou_valid_mask].sum():.2%}; f-mAcc: \t {np.sum(acc_values[acc_valid_mask]*weights_values[acc_valid_mask])/weights_values[acc_valid_mask].sum():.2%}\n')
    confusion = confusion.sum(0) #agregate all scenes statistics
    iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = iou_acc_from_confmat(confusion, num_classes, ignore, mask_nan, verbose, labels)
    metrics = {
        "iou": round(np.mean(iou_values[iou_valid_mask]),3),
        "acc": round(np.mean(acc_values[acc_valid_mask]),3),
        "fiou":round(np.sum(iou_values[iou_valid_mask]*weights_values[iou_valid_mask])/weights_values[iou_valid_mask].sum(), 3),
        "facc":round(np.sum(acc_values[acc_valid_mask]*weights_values[acc_valid_mask])/weights_values[acc_valid_mask].sum(), 3),
    }
    thirds = len(iou_values)//3
    for split, i in [['head',0], ['comm',1], ['tail',2]]:
        idx_i, idx_e = thirds * i,thirds * (i + 1)
        metrics[f"iou_{split}"] = round(np.mean(iou_values[idx_i:idx_e][iou_valid_mask[idx_i:idx_e]]), 3)
        metrics[f"acc_{split}"] = round(np.mean(acc_values[idx_i:idx_e][acc_valid_mask[idx_i:idx_e]]), 3)

    if verbose:
        print(f"\nmIoU: \t {metrics['iou']:.2%}; mAcc: \t {metrics['acc']:.2%}\n ")
        print(f"f-mIoU: \t {metrics['fiou']:.2%}; f-mAcc: \t {metrics['facc']:.2%}\n")
        print()
        if iou_values.shape[0]==51:
            for i, split in enumerate(['head', 'comm', 'tail']):
                idx_i, idx_e = thirds * i,thirds * (i + 1)
                print(f'{split}: \t {metrics[f"iou_{split}"]:.2%}')
                print(f'{split}: \t {metrics[f"acc_{split}"]:.2%}')
                print('---')
        if isinstance(output_path, str):
            output_path = Path(output_path)
        with open(output_path/"statistics.txt", "w") as f:
            f.write(f"label, acc, iou, \n")
            count = 0
            for i in range(len(labels)):
                if i not in ignore:
                    f.write(f"{labels[i]}, {acc_values[count]}, {iou_values[count]}, \n")
                    count +=1

    if verbose:
        plot_metrics(iou_values, acc_values, labels, output_path, ignore)
        plot_confmat(confusion, labels, output_path)
    if return_metrics:
        return metrics, confusion
    return np.mean(iou_values[iou_valid_mask]), confusion
     

def eval_scannetpp_semantic(cfg: Dict[str, Any], top_k: List[int] = [1], verbose: bool =True):
    # Import ScanNet++ path
    sys.path.append("/home/tberriel/Workspaces/semsplat_ws/sem3d/ovoslam/submodules/scannetpp")
    from ovoslam.submodules.scannetpp.semantic.eval.eval_semantic import eval_semantic

    scene_ids = cfg["scene_ids"]

    with open(cfg["classes_file"], "r") as f:
        semantic_classes = f.read().splitlines() 
    num_classes = len(semantic_classes)

    confmats = eval_semantic(scene_ids, cfg["preds_dir"], cfg["gt_dir"], cfg["data_root"],
                            num_classes, -100, top_k, eval_against_gt=False)
    if verbose:
        for k, confmat in confmats.items():
            print(f'Top {k} mIOU: {confmat.miou}')
            
        for class_name, class_iou in zip(semantic_classes, confmat.ious):
            print(f'{class_name: <25}: {class_iou}')

        print('----------------------------------------------------')
    return confmats[1].miou


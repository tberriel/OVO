from __future__ import annotations
from typing import Dict, Any, Tuple, List

import torchvision.transforms.functional as F
from copy import deepcopy
import numpy as np
import heapq
import torch
import os


def mask2segmap(masks: np.ndarray, image: np.ndarray, sort: bool = True) -> Tuple[np.ndarray, np.ndarray] :
    if sort:
        masks = heapq.nlargest(len(masks), masks, key= lambda x: x['stability_score'])  

    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    binary_maps = []
    for i, mask in enumerate(masks):
        binary_maps.append(mask['segmentation'])
        seg_map_mask = mask['segmentation'].copy()
        if sort:
            mask_overlap = np.logical_and(seg_map>-1,seg_map_mask)
            seg_map_mask[mask_overlap] = False# previous masks have higher stability score, if there is overlap, current mask is removed     
        seg_map[seg_map_mask] = i # If seg map mask .sum() == 0 it won't assign anything. 
    #TODO: If masks is empty np.stack will raise an error
    binary_maps = np.stack(binary_maps)
    return seg_map, binary_maps

def segmap2segimg(binary_map: torch.Tensor, image: torch.Tensor, also_bbox: bool, bbox_margin: int = 50, out_l: int = 224) -> torch.Tensor:
    seg_imgs = []
    
    bboxes_xyxy = batched_mask_to_box(binary_map)
    bboxes_xyhw = batched_box_xyxy_to_xywh(bboxes_xyxy)
    # Should add a filter that neither h nor w are 0,
    for i in range(binary_map.shape[0]):
        padded_img = seg_img_from_image(binary_map[i], bboxes_xyhw[i], image, also_bbox, bbox_margin, out_l)
        seg_imgs.append(padded_img)

    seg_imgs = torch.stack(seg_imgs, axis=0) # b,3,H,W

    return seg_imgs

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """From SAM.utils.amg code
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def batched_box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    """From SAM.utils.amg code"""
    box_xywh = box_xyxy # deepcopy(box_xyxy)
    box_xywh[:,2] = box_xywh[:,2] - box_xywh[:,0]
    box_xywh[:,3] = box_xywh[:,3] - box_xywh[:,1]
    return box_xywh

def segmap2bboximg(binary_map: torch.Tensor, image: torch.Tensor,  bbox_margin: int = 50, out_l: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    seg_imgs = []
    bmaps = []
    if len(binary_map)>0:
        bboxes_xyxy = batched_mask_to_box(binary_map)
        bboxes_xyhw = batched_box_xyxy_to_xywh(bboxes_xyxy)
        # Should add a filter that neither h nor w are 0,
        for i in range(binary_map.shape[0]):
            padded_img, bmap = bbox_img_from_image(binary_map[i], bboxes_xyhw[i], image, bbox_margin, out_l)
            seg_imgs.append(padded_img)
            bmaps.append(bmap)
 
        seg_imgs = torch.stack(seg_imgs, axis=0) # b,3,H,W
        bmaps = torch.stack(bmaps, axis=0)

    return seg_imgs, bmaps

def bbox_img_from_image(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor, bbox_margin: int = 50, size: int =224) -> Tuple[torch.Tensor, torch.Tensor] :
    bbox_img = F.resize(get_bbox_img(bbox, image, bbox_margin), (size,size))
    bmap = F.resize(get_bbox_img(bbox, mask[None], bbox_margin), (size,size))
    return bbox_img, bmap

def seg_img_from_image(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor, also_bbox: bool, bbox_margin: int = 50, size: int =224) -> torch.Tensor :
    seg_img = get_seg_img(mask, bbox, image)
    if also_bbox:
        bbox_img = F.resize(get_bbox_img(bbox, image, bbox_margin), (size,size))
        padded_img = torch.concatenate([F.resize(seg_img, (size,size)), bbox_img], axis=0)
    else:
        padded_img = F.resize(pad_img(seg_img), (size,size))
    return padded_img

def get_seg_img(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    x,y,w,h = bbox
    seg_img = torch.zeros((3, h, w), dtype = image.dtype, device=image.device)
    seg_img[:,mask[y:y+h, x:x+w]] = image[..., y:y+h, x:x+w][:,mask[y:y+h, x:x+w]].clone()
    return seg_img

def get_bbox_img(bbox: Tuple[int, int, int ,int], image: torch.Tensor, bbox_margin: int) ->  torch.Tensor:
    x, y, w, h = increase_bbox_by_margin(bbox, bbox_margin)
    bbox_img = image[..., y:y+h, x:x+w].clone()
    return bbox_img

def pad_img(img: torch.Tensor) -> torch.Tensor:
    c, h, w = img.shape
    biggest_side = max(w,h)
    pad = torch.zeros((c, biggest_side, biggest_side), dtype=img.dtype, device = img.device)
    if h > w:
        pad[...,(h-w)//2:(h-w)//2 + w] = img
    else:
        pad[:,(w-h)//2:(w-h)//2 + h,:] = img
    return pad

def increase_bbox_by_margin(bbox: Tuple[int, int, int ,int], margin: int) ->  Tuple[int, int, int ,int]:
    """ # Functino from https://github.com/hovsg/HOV-SG/blob/main/hovsg/utils/sam_utils.py
    Increases the size of a bounding box by the given margin.

    :param bbox: The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param margin: The margin to increase the bounding box size by in pixels.
    :return: The increased bounding box coordinates as a tuple of (x, y, w, h).
    """
    x, y, w, h = bbox
    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2
    # Check if x is negative
    if x < 0:
        w += x
        x = 0

    # Check if y is negative
    if y < 0:
        h += y
        y = 0
    return (x, y, w, h)


def masks_update(*args, **kwargs) -> Tuple[np.ndarray]:
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def filter(keep: torch.Tensor, masks_result) -> List[np.ndarray]:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.7, score_thr: float = 0.1, inner_thr: float = 0.2, **kwargs) -> torch.Tensor:
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    """From SAM.utils.amg code"""
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def load_sam(config: Dict[str, Any], device: str = "cuda") -> SamAutomaticMaskGenerator:
    """ Load SAM or SAM2 model
    """
    sam_version = config.get("sam_version","2.1")

    model_cards = {"vit_b": "vit_b_01ec64.pth", "vit_h": "vit_h_4b8939.pth", "hiera_l": "hiera_large.pt", "hiera_t": "hiera_tiny.pt"}
    sam_encoder = config.get("sam_encoder","hiera_l")
    checkpoint_path =  os.path.join(config["sam_ckpt_path"],f"sam{sam_version}_{model_cards[sam_encoder]}") 
            
    if sam_version == "":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[sam_encoder](checkpoint=checkpoint_path).to(device).eval()
        sam_config = {
            "points_per_side": config.get("points_per_side",32),
            "pred_iou_thresh": config.get("nms_iou_th",0.8),
            "stability_score_thresh": config.get("stability_score_th",0.85),
            "min_mask_region_area": config.get("min_mask_region_area", 100),
        }
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as SamAutomaticMaskGenerator

        model_cfg = os.path.join("configs",f"sam{sam_version}",f"sam{sam_version}_{sam_encoder}.yaml")
        sam = build_sam2(model_cfg, checkpoint_path, device=device, mode="eval", apply_postprocessing=False)
        sam_config = {
        "points_per_side":config.get("points_per_side",32),
        "pred_iou_thresh": config.get("nms_iou_th",0.8),
        "stability_score_thresh": config.get("stability_score_th",0.95),
        "min_mask_region_area": config.get("min_mask_region_area", 0),
        "use_m2m": config.get("use_m2m", False),
        }
        

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **sam_config
    )
    return mask_generator

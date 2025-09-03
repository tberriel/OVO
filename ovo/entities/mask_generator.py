from typing import Dict, Any, Tuple
import numpy as np
import torch
import tqdm
import os

from ..utils import segment_utils

class MaskGenerator:
    """ Initialize SAM backbone, with a given configuration.
    Args:
        - config (Dict[str, Any]): Configuration dictionary specifying hyperparameters and operational settings.
        - scene_name (str, optional): Name of current scene required to load precomputed masks, or save them if precomputing. Default is None.
        - device (str, optional): Device to run SAM. Must be either 'cpu' or 'cuda'. Default is 'cuda'.
    """
    def __init__(self, config: Dict[str, Any], scene_name: str = None, device = "cuda") -> None:
        self.precomputed = config["precomputed"]
        self.config = config
        if scene_name:
            self.masks_path = os.path.join(config["masks_base_path"], scene_name)
        else:
            assert not config.get("precompute", False), "To precompute masks or use precomputed masks \"scene_name\" is required!"
            self.masks_path = ""

        self.nms_iou_th = config.get("nms_iou_th",0.8)
        self.nms_score_th = config.get("nms_score_th",0.7)
        self.nms_inner_th = config.get("nms_inner_th",0.5)

        self.multi_crop = config.get("multi_crop", False)
        self.device = device
        if (self.precomputed or config.get("precompute", False)) and os.path.isdir(self.masks_path):
            print(" {} path already exists, skipping masks precompute! To recompute masks delete old masks!".format(self.masks_path))
            self.mask_generator = None
        else:
            #import sys
            #torch.compiler.allow_in_graph(sys._getframe)
            self.load_mask_generator(self.config)

    def load_mask_generator(self, config: Dict[str, Any]) -> None:
        sam_version = config.get("sam_version", "")
        if sam_version == "":
            self.dtype = torch.float32
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.dtype = torch.bfloat16
        print("Loading SAM{}".format(sam_version))
        self.mask_generator = segment_utils.load_sam(config, device = self.device)

        with torch.no_grad() and torch.autocast(device_type=self.device, dtype=self.dtype):
            #First pass for compilation
            self.mask_generator.generate(np.random.rand(512,512,3).astype(np.float32))
            self.mask_generator.generate(np.random.rand(512,512,3).astype(np.float32)) 

    def to(self, device: str) -> None:
        """
        Move predictor model to specified device.
        Args:
            device (str): device to mode the model to.
        """
        self.device = device
        if self.mask_generator:
            self.mask_generator.predictor.model.to(device)

    def cpu(self) -> None:
        """
        Move predictor model to cpu device.
        """
        self.device = "cpu"
        if self.mask_generator:
            self.mask_generator.predictor.model.cpu()

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        if self.mask_generator:
            self.mask_generator.predictor.model.cuda()
    
    def get_masks(self, image: np.ndarray, frame_id: int = None):
        """
        Generate or load segmentation and binary masks for a given image.

        Parameters:
            - image (np.ndarray): The input image for which masks are to be generated or loaded.
            - frame_id (int): The frame identifier used to load precomputed masks (optional).

        Returns:
            - seg_map (torch.Tensor): The segmentation maps on self.device.
            - binary_maps (torch.Tensor): The binary maps on self.device.
        """

        if self.precomputed:
            seg_map, binary_maps = self._load_masks(frame_id)
        else:
            seg_map, binary_maps = self.segment(image)

        return torch.from_numpy(seg_map).to(self.device), torch.from_numpy(binary_maps).to(self.device)
    
    @torch.no_grad
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        For an image compute SAM masks and conver to binary maps.
        Args:
            - image (np.ndarray): (H, W, 3) shape RGB image.
        Return:
            - seg_map (np.ndarray): Segmentation map of shape (H,W) with pixel values in [-1, N), where each pixel value indicates the id of on of the N segmentation mask. Unasigned pixels have value -1.
            - binary_maps (np.ndarray): array of shape (N, H, W). Each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
        """

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            masks = self.mask_generator.generate(image) 
            if len(masks)==0:
                return np.array([]), np.array([])
            
            masks_default, = segment_utils.masks_update(masks, iou_thr=self.nms_iou_th, score_thr=self.nms_score_th, inner_thr=self.nms_inner_th)
            seg_map, binary_maps = segment_utils.mask2segmap(masks_default, image)
        
        return seg_map, binary_maps
    
    def precompute(self, dataset: torch.utils.data.Dataset, segment_every: int) -> None:
        """
        Precomputes segmentation masks for the given dataset and frame IDs.
        This method checks if the segmentation masks for the specified frame IDs
        already exist. If they do not, it generates the masks using the mask
        generator and saves them to the specified path.
        Args:
            - dataset (Dataset): The dataset from which to generate segmentation masks.
            - segment_every (int): How many frames to skip between semantic segmentations.
        """
        print("Precomputing segmentation masks.")
        
        os.makedirs(self.masks_path, exist_ok=True)

        segmenting_frame_ids = [i for i in range(len(dataset)) if i%segment_every == 0]
        frame_ids = tqdm.tqdm(segmenting_frame_ids)
        for frame_id in frame_ids:

            map_path = os.path.join(self.masks_path, f"{frame_id:04d}_seg_map_default.npy") 
            bmap_path = os.path.join(self.masks_path, f"{frame_id:04d}_bmap_default.npy") 

            if os.path.exists(map_path) and os.path.exists(bmap_path):
                print(f"Frame {frame_id} already compute. Skipping ...") 
            else:
                if self.mask_generator is None:
                    self.load_mask_generator(self.config)
                seg_maps, binary_maps = self.segment(dataset[frame_id][1])
                self._save_masks(seg_maps, binary_maps, frame_id)

        self.precomputed = True    
    
    def _save_masks(self, seg_map: np.ndarray, binary_maps: np.ndarray,  frame_id: int) -> None:
        """
        Saves the segmentation map and binary maps of a fiven frame to self.masks_path.
        Args:
            - seg_map (np.ndarray): The segmentation map to be saved.
            - binary_maps (np.ndarray): The binary maps to be saved.
            - frame_id (int): The frame identifier used to name the saved files.
        """

        map_path = os.path.join(self.masks_path, f"{frame_id:04d}_seg_map_default") 
        np.save(map_path, seg_map)

        bmap_path = os.path.join(self.masks_path, f"{frame_id:04d}_bmap_default") 
        np.save(bmap_path, binary_maps)
            
        return
    
    def _load_masks(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads segmentation masks for a given frame ID.
        Args:
            frame_id (int): The ID of the frame for which to load the segmentation images.
        Returns:
           -  seg_map (np.ndarray): Segmentation map of shape (H,W) with pixel values in [-1, N), where each pixel value indicates the id of on of the N segmentation mask. Unasigned pixels have value -1.
            - binary_maps (np.ndarray): array of shape (N, H, W). Each pixel will have a value of 1 if it belongs to corresponding segmentation mask, or 0 otherwise.
        """

        map_path = os.path.join(self.masks_path, f"{frame_id:04d}_seg_map_default.npy") 
        if os.path.exists(map_path):
            seg_map = np.load(map_path)
            
            #TODO: store and load original binary maps rather than compute them from the saved filtered seg map
            binary_path = os.path.join(self.masks_path, f"{frame_id:04d}_bmap_default.npy")
            if os.path.exists(binary_path):
                binary_maps = np.load(binary_path)
            else:
                idxs = np.arange(seg_map.max()+1)
                binary_maps = np.repeat(np.expand_dims(seg_map, 0),len(idxs),axis=0) == np.expand_dims(idxs,[-1,-2])
        else:
            print(f"No precomputed mask for frame {frame_id}")
            seg_map, binary_maps = np.array([]), np.array([])

        return seg_map, binary_maps
    
from typing import Any, Dict, List, Tuple
from collections import deque
import numpy as np
import pprint
import torch
import time

from ..utils import geometry_utils
from .clip_generator import CLIPGenerator
from .mask_generator import MaskGenerator
from .instance3d import Instance3D
from .logger import Logger

class OVO:
    """ Initialize CLIP and SAM backbones, with a given configuration, and logger.
    Args:
        - config (Dict[str, Any]): Configuration dictionary specifying hyperparameters and operational settings.
        - logger (Logger): Object for logging the tracking process.
        - scene_name (str, optional): Name of current scene required to load precomputed masks, or save them if precomputing. Default is None.
        - cam_intrinsics (torch.Tensor, optional): Camera intrinsic matrix. Required for scene reconstruction but not for evaluation. Default is None.
        - eval (bool, optional): If True SAM backbone is not loaded and the Camera intrinsic matrix is not required. Default is False.
        - device (str, optional): Device to run CLIP model and SAM. Must be either 'cpu' or 'cuda'. Default is 'cuda'.
    """
    def __init__(self, config: Dict[str, Any], logger: Logger, scene_name: str | None = None, cam_intrinsics: torch.Tensor | None = None, eval: bool = False, device = "cuda") -> None:
        if not eval:
            assert cam_intrinsics is not None, "Camera intrinsics required for reconstruction!"

        config["sam"]["multi_crop"] = False if config["clip"]["embed_type"] == "vanilla" else True
        self.cam_intrinsics = cam_intrinsics
        self.config = config
        self.logger = logger
        self.debug_info = config.get("debug_info", False)
        self.device = device
        self.n_top_views = config["clip"].get("k_top_views", 0)
        Instance3D.n_top_kf = self.n_top_views

        self.clip_generator = CLIPGenerator(config["clip"], device=device)
        if not eval:
            self.mask_generator = MaskGenerator(config["sam"], scene_name, device=device)
        else:
            self.mask_generator = None
        self.keyframes = {
            "ins_descriptors": dict(),
            "frame_id": list(),
            "obj_maps": list(),
        }
        self.keyframes_queue = deque([])
        self.objects=dict()
        self._time_cache = []
        
        self.next_ins_id = 0
        self.kf_id = 0

        print('Semantic config')
        pprint.PrettyPrinter().pprint(config)


    def to(self, device: str) -> None:
        """
        Move predictor model to either 'cpu' or 'cuda' device.
        Args:
            device (str): device to mode the model to.
        """
        if "cuda" in device:
            return self.cuda()
        else:
            return self.cpu()

    def cpu(self) -> None:
        """
        Move predictor model to cpu device.
        """
        self.device = "cpu"
        self.clip_generator.cpu()
        if self.mask_generator is not None:
            self.mask_generator.cpu()

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        self.clip_generator.cuda()
        if self.mask_generator is not None:
            self.mask_generator.cuda()

    def profil(func):
        """A decorator that profiles functions running time if self.config["log"] == True.
        Args:
            - func: The function to be decorated.
        Returns:
            - The wrapper function.
        """
        def wrapper(self, *args, **kwargs):
            if self.config.get("log", False):
                torch.cuda.synchronize()
                start_time = time.time()
                out = func(self, *args, **kwargs)
                torch.cuda.synchronize()
                end_time = time.time()
                self._time_cache.append(end_time-start_time)
                return out
            else:
                return func(self, *args, **kwargs)
        return wrapper    
    
    def detect_and_track_objects(self, frame_data: Tuple[int, np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], c2w: torch.Tensor) -> torch.Tensor:
        """ For the current frame (1) computes using SAM for each level i \in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance. 

        Args:
            - frame_data (tuple): current frame data.
                - frame_id (int): current frame id.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - points_3d (torch.Tensor): set of 3D point coordinates to match to current fram segmentation maps.
                - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
                - points_ins_ids (torch.Tensor): ids of objects associated to each 3d point for each segmentation level of previous keyframes.
            - c2w: (torch.Tensor): camera to world 3D transform.
        Update: 
            - points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point after current keyframe segmentation.
        """
        frame_id, image = frame_data[:2]

        seg_maps, binary_maps = self._get_masks(image, frame_id)
        if len(seg_maps) == 0:
            print(f"No mask segmented in {frame_id}!")
            return None

        last_id = self.next_ins_id
        matched_ins_ids, binary_maps, n_matched_points = self._match_and_track_instances(frame_data[1:], map_data, c2w, seg_maps, binary_maps)
            
        # Save keyframe information
        self.keyframes_queue.append([matched_ins_ids, binary_maps, image, self.kf_id])
        self.kf_id +=1

        if self.config.get("log", False):
            self.keyframes["frame_id"].append(frame_id)
            self.logger.log_ovo_stats(
                {
                    "frame_id":frame_id,
                    "n_obj":[self.next_ins_id-last_id],
                    "n_matches":n_matched_points, 
                    "t_sam":round(self._time_cache[0],2),
                    "t_obj":round(self._time_cache[1],3),
                },
                print_output=True
                )
            self._time_cache = []
    
    @profil
    def _get_masks(self, image: np.ndarray, frame_id: int):
        """ Profiled call to mask_generator to either compute segmentation maps for image, or load precomputed segments.
        Args:
            - frame_id (int): current frame id.
            - image (np.ndarray): RGB image with shape (H, W, 3).

        Returns:
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): The binary maps on self.device with shape (N, H, W).
        """
        return self.mask_generator.get_masks(image, frame_id)
    
    @profil
    def _match_and_track_instances(self, frame_data: Tuple[int, np.ndarray, np.ndarray, Tuple[float, float, int]], map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], c2w: torch.Tensor, seg_map: torch.Tensor, binary_maps: torch.Tensor) -> Tuple[List[int], torch.Tensor, int]:
        """ For the current frame (1) computes using SAM for each level i \in M, a set of segmentation maps; (2) track segmentation maps between frames projecting 3D points and associating the map to 3D instances, if 3D points don't have an associated 3D instance, create a new; (3) associate 3D points without an instance id to matched instances; (4) fuse 2D segments associated to the same 3D instance. 

        Args:
            - frame_data (tuple): current frame data.
                - image (np.ndarray): RGB image with shape (H, W, 3).
                - depth (np.ndarray): Frame depth with shape (h, w).
                - rgb_depth_ratio (tuple): If H == h and W == w, tuple is empty, otherwise stores (r_h, r_w, crop_edge), such that H = (h+2*crop_edge)*r_h, and W = (w+2*crop_edge)*r_w
            - map_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - points_3d (torch.Tensor): set of 3D point coordinates to match to current fram segmentation maps.
                - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
                - points_ins_ids (torch.Tensor): ids of objects associated to each 3d point for each segmentation level of previous keyframes.
            - c2w: (torch.Tensor): camera to world 3D transform.
            - seg_map (torch.Tensor): The segmentation maps on self.device with shape (H, W).
            - binary_maps (torch.Tensor): Tensor of shape (N, H, W) on self.device, where each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
        Update: 
            - points_ins_ids (torch.Tensor): updated ids of 3D instances associated to each 3d point with current keyframe segmentation.
        Return:
            - matched_ins_ids (List): Ids of 3D instances matched in current frame 
            - binary_maps (torch.Tensor): The matched binary maps on self.device with shape (M, H, W).
            - n_matched_points (int): numbed of 3D points matched with 3D instances in current keyframe.
        """
        kf_id = self.kf_id
        image, depth, rgb_depth_ratio = frame_data
        points_3d, points_ids, points_ins_ids = map_data

        depth = torch.from_numpy(depth).to(self.device)        
        camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
        frustum_mask = geometry_utils.compute_frustum_point_ids(points_3d, camera_frustum_corners, device=self.device)
        frustum_points_3d = points_3d[frustum_mask]

        if self.config.get("depth_filter", False):
            depth = geometry_utils.depth_filter(depth)

        matched_points_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(depth, torch.linalg.inv(c2w), frustum_points_3d, self.cam_intrinsics, self.config["match_distance_th"])

        if len(rgb_depth_ratio)>0:
            matches += rgb_depth_ratio[-1]
            matches[:,1] = (matches[:,1]*rgb_depth_ratio[0]).int()
            matches[:,0] = (matches[:,0]*rgb_depth_ratio[1]).int()
        matched_seg_idxs = seg_map[matches[:,1], matches[:,0]]

        frustum_points_ids, frustum_points_ins_ids = points_ids[frustum_mask], points_ins_ids[frustum_mask]
        frustum_points_ins_ids, matched_ins_info = self._track_objects(frustum_points_ids, frustum_points_ins_ids, matched_points_idxs, matched_seg_idxs, seg_map, self.config["track_th"], kf_id)
        matched_ins_ids, binary_maps = self._fuse_masks_with_same_ins_id(binary_maps, matched_ins_info, kf_id)

        points_ins_ids[frustum_mask] = frustum_points_ins_ids # Updates points_ins_ids

        if self.config.get("debug_info", False):
            ins_maps = torch.ones(image.shape[:2], dtype=torch.int, device=self.device)*-1
            for ins_id, matches in matched_ins_info.items():
                for map_idx, _ in matches:
                    ins_maps[binary_maps[map_idx]] = ins_id
            self.keyframes["ins_maps"].append(ins_maps.cpu().numpy())

        return matched_ins_ids, binary_maps, len(matched_points_idxs)
            
    def _track_objects(self, points_ids: torch.Tensor, points_ins_ids: torch.Tensor, matched_points_idxs: torch.Tensor, matched_seg_idxs: torch.Tensor, seg_map: torch.Tensor, track_th: float, kf_id: int) -> tuple[torch.Tensor, Dict[int, List[Tuple[int, int]]]]:
        """  We project 3D points and match with segmentation maps. Then we assign to each segmentation map the id of the 3D instance associated with the majority of points projected into it. If the set points don't have an object assigned, a new object is created and assigned to them. Points without an object assigned get assigned the segmentation map's instance.
        Args:
            - points_ids (torch.Tensor): ids to identify 3D points in case their order changes, or any of them is pruned, between keyframes.
            - points_ins_ids (torch.Tensor): ids of 3D instances associated to each 3d point.
            - matched_points_idxs (torch.Tensor): idxs in points_3d of N matched points.
            - matched_seg_idxs (torch.Tensor): (N) tensor of the indexes of the segmentation map matched to each of N 3D points. 
            - seg_map (torch.Tensor): (H,W) tensor where each pixel stores the idx of the corresponding mask.
            - kf_id (int): current keyframe id.
        Return:
            - points_ins_ids (torch.Tensor): ids of 3D instances associated to each element of points_ids.
            - matched_ins_info (Dict[int, List[Tuple[int, int]]]]): Hash map storing for each observed 3D instance, a list of (matched mask index, mask area).
        """

        matched_ins_info = {}
        for map_idx in range(seg_map.max()+1):
            map_ins_id = -1
            map_points = matched_points_idxs[matched_seg_idxs == map_idx]
            if len(map_points)> track_th:
                mask_area = (seg_map == map_idx).sum().item()
                assigned_mask = points_ins_ids[map_points] > -1                    
                unassigned_points_ids = points_ids[map_points[~assigned_mask]].cpu().tolist()
                #Assign points to 3D instance, or create a new instance
                if assigned_mask.sum().item() > track_th:
                    map_ins_id = torch.mode(points_ins_ids[map_points[assigned_mask]]).values.item()
                    self.objects[map_ins_id].update(unassigned_points_ids, kf_id, mask_area)
                    if map_ins_id in matched_ins_info.keys():
                        matched_ins_info[map_ins_id].append((map_idx, mask_area))  
                    else:
                        matched_ins_info[map_ins_id]=[(map_idx, mask_area)]

                elif len(unassigned_points_ids) > track_th:                    
                    map_ins_id = self.next_ins_id
                    self.next_ins_id +=1
                    #assigned points do not change obj id
                    self.objects[map_ins_id] = Instance3D(map_ins_id, kf_id=kf_id, points_ids=unassigned_points_ids, mask_area=mask_area)
                    matched_ins_info[map_ins_id]=[(map_idx, mask_area)]

            if map_ins_id > -1:
                # Assignto matched unassigned points (id==-1) new instance id 
                points_ins_ids[map_points[~assigned_mask]] = map_ins_id
        
        return points_ins_ids, matched_ins_info

    def _fuse_masks_with_same_ins_id(self, binary_maps: torch.Tensor, matched_ins_info: Dict[int, List[Tuple[int, int]]], kf_id: int) -> Tuple[List[int], torch.Tensor] :
        """ A 3D object can be mapped to more than one 2D mask. We fuse masks that belong to the same ins_id, keeping idx of first occurence. Objects matched to fused masks are updated to the new masks areas.
        Args:
            - binary_maps (torch.Tensor): Tensor of shape (N, H, W) on self.device, where each pixel will have a value of 1 if it belongs to the nth segmentation mask, or 0 otherwise.
            - matched_ins_info (dict): Hash map storing for each observed 3D instance, a list of (matched mask index, mask area).
            - kf_id (int): current keyframe id.
        Return:
            - matched_ins_ids:
            - binary_maps (torch.Tensor): Updated binary maps on self.device with shape (M, H, W).
        """

        matched_ins_ids = []
        maps_idxs=[]
        for ins_id, data_list in matched_ins_info.items():
            map_idx = data_list[0][0]
            if len(data_list)>1:
                for j in range(1,len(data_list)):
                    binary_maps[map_idx] = torch.logical_or(binary_maps[map_idx], binary_maps[data_list[j][0]])
                    
                mask = binary_maps[map_idx]
                mask_area = mask.sum().item()

                if self.n_top_views>0:
                    self.objects[ins_id].add_top_kf(kf_id, mask_area)

            if self.n_top_views<=0 or self.objects[ins_id].is_top_kf(kf_id):
                matched_ins_ids.append(ins_id)
                maps_idxs.append(map_idx)

        binary_maps = binary_maps[maps_idxs]

        return matched_ins_ids, binary_maps

    def compute_semantic_info(self) -> None:
        if len(self.keyframes_queue)>self.config.get("kf_queue_delay", 0):
            self._compute_semantic_info() 

    def complete_semantic_info(self) -> None:
        while len(self.keyframes_queue)>0:
            self._compute_semantic_info()
    
    def _compute_semantic_info(self) -> None:
        """ Compute semantic information of first keyframe in the queue.
        """
        matched_ins_ids, binary_maps, image, kf_id = self.keyframes_queue.popleft()
        
        if len(matched_ins_ids)>0:
            if self.n_top_views > 0:
                obj_to_compute = []
                for j, ins_id in enumerate(matched_ins_ids):
                    if self.objects[ins_id].is_top_kf(kf_id):
                        obj_to_compute.append(j)
                if len(obj_to_compute) == 0:
                    return
                matched_ins_ids, binary_maps = np.asarray(matched_ins_ids)[obj_to_compute].tolist(), binary_maps[obj_to_compute]


            image = torch.from_numpy(image.transpose((2,0,1))).to(self.device)
            seg_images = self._get_seg_image(image, binary_maps)
            clip_embeds = self._extract_clip(image[None,...]/255., seg_images/255.).cpu()
            self._update_matched_objects_clip(clip_embeds, matched_ins_ids, kf_id)

            if self.config.get("log", False):
                frame_id = self.keyframes["frame_id"][kf_id]
                self.logger.log_ovo_stats(
                    {
                    "frame_id":frame_id,
                    "t_seg": round(self._time_cache[0],2),
                    "t_clip": round(self._time_cache[1],2),
                    "t_up": round(self._time_cache[2],3)
                    }
                    ,
                    print_output=True
                    )
                self._time_cache = []
                
    @profil
    def _get_seg_image(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """Profiled call to self.mask_generator.get_seg_img

        Args:
            - binary_maps (tensor): A tensor of (N, H, W) containing N binary maps.
            - image (tensor): A tensor of shape (H, W, 3) representing the input image.

        Returns:
            - seg_images (torch.Tensor): Segmented images with shape (N, 3, h, w), if self.config["clip"]["embed_type"] == "vanilla" , else (N, 6, h, w) with h = w = self.config["sam"]["mask_res"].
        """
        return self.mask_generator.get_seg_img(binary_maps, image)
    
    @profil
    def _extract_clip(self, image: torch.Tensor, seg_images: torch.Tensor) -> List[Any]:
        """Profiled call to self.clip_generator.extract_clip. Computes a CLIP vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (H,W,3) and range 0-255.
            - seg_images (torch.Tensor): array of shape (N,6,h,w), with h < H and w < W. The first 3 channels of the second dimension store the segment with black background of a 2D instance, while the last 3 channels store the image of the minimum bounding box arround that 2D semgent with background.
            - return_all: if True returns the three computed descriptors of each image in seg_images instead of merging them.
        Return:
            - climp_embeds: each level/key stores a list of numpy arrays with dim (N, self.clip_dim).    
        """
        return self.clip_generator.extract_clip(image, seg_images, self.config.get("return_all_clips", False))

    @profil
    def _update_matched_objects_clip(self, clip_embeds: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:  
        """
        Store clip_embeds keyframe information, and updates matched 3D instances' clip embeddings.
        Args:
            - clip_embeds (torch.Tensor): A tensor containing the clip embeddings.
            - matched_ins_ids (List[int]): A list of instance IDs that are matched with the clip embeddings.
            - kf_id (int): current keyframe id.
        Updates:
            self.keyframes["ins_descriptors"]
        """

        ins_embeds = dict()
        for i, ins_id in enumerate(matched_ins_ids):
            if ins_id != -1:
                ins_embeds[ins_id] = clip_embeds[i]

        # Save keyframe information
        self.keyframes["ins_descriptors"][kf_id] = ins_embeds
        
        for ins_id in matched_ins_ids:
            self.objects[ins_id].update_clip(self.keyframes["ins_descriptors"])
        return
    
    def update_objects_clip(self) -> None:
        """ Update all 3D instances descriptors
        """
        for object in self.objects.values():
            object.update_clip(self.keyframes["ins_descriptors"])
        return
    
    @torch.no_grad()
    def classify_instances(self, classes: List[str], template: str | List[str] = "This is a photo of a {}", th: float = 0):
        """
        Classifies 3D instances based on provided classes and templates.
        Args:
            - classes (List[str]): A list of class names to classify the instances into.
            - templates (str | List[str], optional): A template or a list of templates to use for classification. If it's a list, the classes embeddings will be an ensembles of the templates. 
            - th (float, optional): Minimum confidence to classify an instance. If highest confidenc is lower than th, the instance remains unclassified (-1). Default 0.
        Returns:
            dict: A dictionary containing:
                - "classes" (numpy.ndarray): An array of classes indices corresponding to each 3D instance.
                - "conf" (numpy.ndarray): An array of confidence scores for each classification.
        """

        sim_map = self.query(classes, template)
        instances_classes = torch.argmax(sim_map,dim=1)
        max_conf = torch.gather(sim_map, -1, instances_classes[:,None]).squeeze()
        instances_classes[max_conf <=th] = -1
        max_conf[max_conf<=th] = 0
        instances_info = {"classes": instances_classes.cpu().numpy(), "conf":max_conf.cpu().numpy()}
        return instances_info
    
    @torch.no_grad()
    def query(self, queries: List[str], templates: List[str] = ['{}'], ensemble: bool = False) -> torch.Tensor:
        """
        Queries the 3D instances using the provided queries and templates.
        Args:
            - queries (List[str]): A list of query strings to be used for querying the 3D instances.
            - templates (List[str], optional): A list of template strings to format the queries. Defaults to ['{}'].
            - ensemble (bool, optional): A flag indicating whether to use ensemble method for querying. Defaults to False.
        Returns:
            - torch.Tensor: A relevance map tensor of shape (len(queries), n_objs) indicating the similarity between the queries and the 3D instances.
        Raises:
            AssertionError: If there are no 3D instances to query.
        """
        assert len(self.objects) > 0, "No 3D instances to query!"
        obj_clips = self.get_objs_clips()
        relev_map = self.clip_generator.get_embed_txt_similarity(obj_clips.to(self.device), queries, templates=templates)
        return relev_map

    @torch.no_grad()
    def get_objs_clips(self) -> torch.Tensor:
        """ Retrieve all N 3D instances' descriptors.
        Return:
            - torch.Tensor: A tensor with shape (N, self.clip_generator.clip_dim) on self.device.
        """    
        object_clips = torch.zeros((len(self.objects), self.clip_generator.clip_dim), device = self.device)
        for j, obj in enumerate(self.objects.values()):
            if obj.clip_feature is not None:
                object_clips[j] = obj.clip_feature.to(self.device)
        return object_clips    

    def capture_dict(self, debug_info: bool) -> Dict[str, Any]:
        """
        Captures the current state of the scene and returns it as a dictionary.
        Args:
            debug_info (bool): If True, includes additional debug information in the dictionary.
        Returns:
            dict: A dictionary containing the current state of the scene. If debug_info is True,
                  the dictionary will also include frame IDs, default object maps, and object descriptors.
        """
        scene_dict = {
            "ins_3d_ids": np.asarray(list(self.objects.keys()))
        }
        for obj in self.objects.values():
            scene_dict.update(obj.export(debug_info))
        if debug_info:
            scene_dict["frame_id"] = np.array(self.keyframes["frame_id"])
            scene_dict["ins_map"] = np.array(self.keyframes["ins_maps"])
            for kf_id, ins_descriptors in self.keyframes["ins_descriptors"].items():
                for ins_id, descriptors in ins_descriptors.items():
                    scene_dict[f"kf_{kf_id}_ins3d_{ins_id}_clips"] = descriptors.cpu().numpy()
        return scene_dict

    def restore_dict(self, scene_dict: Dict[str, Any], debug_info: bool = False): 
        """
        Restores the state of the object from a given scene dictionary.
        Args:
            scene_dict (Dict[str, Any]): A dictionary containing the scene data to restore.
            debug_info (bool, optional): If True, additional debug information will be restored. Defaults to False.
        Raises:
            Exception: Catches and ignores any exceptions during the restoration process.
        Notes:
            - Iterates through the keys of the scene_dict to restore Instance3D instances.
            - If debug_info is True, restores keyframes information including frame IDs, instance maps, and instance descriptors.
        """

        for i in scene_dict["ins_3d_ids"]:
            obj = Instance3D(i)
            obj.restore(scene_dict, debug_info)
            self.objects[obj.id] = obj
        if debug_info:
            self.keyframes["frame_id"] = list(scene_dict["frame_id"])
            self.keyframes["ins_maps"] = [x.squeeze() for x in np.split(scene_dict["ins_map"], len(self.keyframes["frame_id"]))]
            for i in range(len(self.keyframes["frame_id"])):
                self.keyframes["ins_descriptors"][i] = {}
                for ins_id in self.objects.keys():
                    descriptor = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_clips", None)
                    if descriptor is not None:
                        self.keyframes["ins_descriptors"][i][ins_id] = torch.tensor(descriptor, device=self.device)
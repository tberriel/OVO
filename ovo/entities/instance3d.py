
from typing import Any, Dict, List
import numpy as np
import torch
import heapq
class Instance3D:
    """
    3D instance class.

    Args:
        id (int): The unique identifier for the 3D instance.
        kf_id (int, optional): Keyframe ID where the instance has been observed. Defaults to None.
        points_ids (List[int], optional): A list of point IDs associated with the object. Defaults to None.
        mask_area (int, optional): The area of the first mask associated with the object. Defaults to 0.
        n_top_kf (int, optional): The number of top keyframes to keep. Defaults to -1.

    Attributes:
        id (int): The unique identifier for the object.
        clip_feature (None): Placeholder for clip feature.
        clip_feature_kf (None): Placeholder for clip feature keyframe.
        kfs_ids (list): A list of keyframe IDs associated with the object.
        points_ids (list): A list of point IDs associated with the object.
        to_update (bool): Flag indicating if the object descriptor needs to be updated.
        n_top_kf (int): The number of top keyframes to keep. If 0 all keyframes are used.
        top_kf (list): A Heap of top keyframes ordered by their mask area.
    """
    n_top_kf: int = 0

    def __init__(self, id: int, kf_id: int | None = None, points_ids: List[int] = None, mask_area: int = 0):
        self.id = id
        self.clip_feature = None
        self.clip_feature_kf = None
        self.kfs_ids = []
        self.points_ids = []
        self.top_kf = []
        self.to_update = False
        if kf_id is not None:
            self.update(points_ids, kf_id, mask_area)

    def update(self, points_ids: List[int], kf_id: int, area: int) -> None:
        """ Add repeated
        Args:
            - points_id (List[int]): ids of points matched with the object.
            - kf_id (int): id of keyframe where the object has been observed.  
            - area (int): area of the segmentation map on current keyframe
        Return:
            - True if current KeyFrame is in the top_k view, False otherwise
        """
        self.add_keyframes(kf_id)
        self.add_points_ids(points_ids)
        self.add_top_kf(kf_id, area)

    def add_points_ids(self, points_ids: List[int]) -> None:
        """Add points_ids to list of points matched with the object. 
        Args:
            - points_id (List[int]): ids of points matched with the object.
        """
        self.points_ids.extend(points_ids)

    def add_keyframes(self, kf_id: int) -> None:
        """If frame  no already in list, add to list of keyframes where the object has been observed.
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed. 
        """
        if kf_id not in self.kfs_ids:  
            self.kfs_ids.append(kf_id)

    def add_top_kf(self, kf_id: int, area: int)-> None:
        """ If self.n_top_kf <=0, the self.to_update is set to True but self.top_kf remains empty. Otherwise, if the KF is already on the list of best, update area value. Else, if the area is one of the N biggest, add to list of top keyframes, in both cases self.to_update is set to True.
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed.  
            - area (int): area of the segmentation map on current keyframe
        Return:
            - True if current KeyFrame is in the top_k view, False otherwise
        """        
        idx = self.idx_in_top_kf(kf_id)
        if idx > -1 :
            if area > self.top_kf[idx][0]:
                self.top_kf[idx] = (area, kf_id)
                heapq.heapify(self.top_kf)
                self.to_update=True
        else:
            self._add_top_kf(kf_id, area)
    
    def _add_top_kf(self, kf_id: int, area: int) -> None:
        """If the area is one of the N biggest, add to list of top keyframes
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed. 
            - area (int): area of the segmentation map of the object if keyframe_id 
        """
        if len(self.top_kf) < self.n_top_kf:
            heapq.heappush(self.top_kf,(area, kf_id))
            self.to_update=True
        else:
            removed = heapq.heappushpop(self.top_kf,(area, kf_id))
            if (self.n_top_kf <= 0) or (removed[1] != kf_id):
                self.to_update = True
       
    def idx_in_top_kf(self, kf_id: int) -> int:
        """ If kf_id is in self.top_kf, returns the index. Otherwise return -1.
        Args:
            - kf_id (int): id of keyframe to search.
        Return type:
            - (int)
        """
        for idx, (_, id) in enumerate(self.top_kf):
            if id == kf_id:
                return idx
        return -1        
    
    def is_top_kf(self, kf_id: int) -> bool:
        """ If kf_id is in self.top_kf, returns True. Otherwise False.
        Args:
            - kf_id (int): id of keyframe to search.
        Return type:
            - (bool)
        """
        return self.idx_in_top_kf(kf_id) > -1     

    def update_clip(self, keyframes_clips: Dict[int, Dict[int, torch.Tensor]], force_update: bool = False) -> None:
        """If self.to_update is True, compute CLIP vector minimizing L1 norm of associated clip vectors from keyframes where the object was observed. Minimizing the L1 norm is equivalent to compute the median of the vector's norm.
        Args:
            - keyframes_clips (Dict[int, Dict[int, torch.Tensor]]): for each keyfram store a dictionary where the keys are object ids and values are associated clip vectors.
            - force_update (bool): if True, recomputed Instance3D descriptors even self.to_update == False
        Updates:
            - self.clip_feature
            - self.clip_feature_kf
            - self.to_update
        """
        if self.to_update or force_update:
            clips = []
            if self.n_top_kf > 0:
                for _, kf in heapq.nlargest(self.n_top_kf,self.top_kf):
                    kf_clips = keyframes_clips.get(kf)
                    if kf_clips is not None:
                        clips.append(kf_clips[self.id])
            else:
                for kf in self.kfs_ids:
                    kf_clips = keyframes_clips.get(kf)
                    if kf_clips is not None:
                        clips.append(kf_clips[self.id])
                
            if len(clips) == 0:
                return
            
            clips = torch.vstack(clips)
            clips = clips[:,None]
            l1_distances = torch.abs(clips-clips.permute(1,0,2)).sum((1,2))
            kf = l1_distances.argmin()
            self.clip_feature = clips[kf]
            self.clip_feature_kf = kf
            self.to_update = False

    def export(self, debug_info: bool = False) -> Dict[str, Any]:
        """Export object properties as a dictionary.
        Args:
            debug_info (bool): If True, includes additional debug information (self.kfs_ids, self.points_ids, self.top_kf) in the dictionary.
        Returns:
            dict: A dictionary containing the current state of the Instance. If debug_info is True,
                  the dictionary will also include keyframe IDs, point ids and top keyframes ids and areas.
        """
        obj_dict = {
            f"ins3d_{self.id}_clip_feature": self.clip_feature,
            f"ins3d_{self.id}_clip_feature_kf": self.clip_feature_kf,
        }

        if debug_info:
            obj_dict.update({
            f"ins3d_{self.id}_keyframes_ids":  np.array(self.kfs_ids),
            f"ins3d_{self.id}_points_ids":  np.array(self.points_ids),
            f"ins3d_{self.id}_top_kfs":  np.array(self.top_kf),
            })

        return obj_dict
    
    def restore(self, obj_dict: Dict[str, Any], debug_info: bool) -> None:
        """Restore object properties from a dictionary.
        Args:
            obj_dict (Dict[str, Any]): A dictionary containing the current state of the Instance. If debug_info is True,
                  the dictionary must also include keyframe IDs, point ids and top keyframes ids and areas.
            debug_info (bool): If True, expected additional debug information in the dictionary.
        """
        self.clip_feature = obj_dict[f"ins3d_{self.id}_clip_feature"]
        self.clip_feature_kf = obj_dict.get(f"ins3d_{self.id}_clip_feature_kf", None)
        self.to_update=self.clip_feature is None
        if debug_info:
            self.kfs_ids = obj_dict[f"ins3d_{self.id}_keyframes_ids"].tolist()
            self.points_ids = obj_dict[f"ins3d_{self.id}_points_ids"].tolist()
            if obj_dict.get(f"ins3d_{self.id}_top_kfs", None) is not None:
                self.top_kf=[(area,kf_id) for area, kf_id in obj_dict[f"ins3d_{self.id}_top_kfs"]]

    def old_restore(self, obj_dict: Dict[str, Any], debug_info: bool) -> None:
        """Restore object properties from a dictionary.
        Args:
            obj_dict (Dict[str, Any]): A dictionary containing the current state of the Instance. If debug_info is True,
                  the dictionary must also include keyframe IDs, point ids and top keyframes ids and areas.
            debug_info (bool): If True, expected additional debug information in the dictionary.
        """
        self.clip_feature = torch.tensor(obj_dict[f"default_{self.id}_clip_feature"])
        self.clip_feature_kf = obj_dict.get(f"default_{self.id}_clip_feature_kf", None)
        if debug_info:
            self.kfs_ids = obj_dict[f"default_{self.id}_keyframes_ids"].tolist()
            self.points_ids = obj_dict[f"default_{self.id}_points_ids"].tolist()
            if obj_dict.get(f"default_{self.id}_top_kfs", None) is not None:
                self.top_kf=[(area,kf_id) for area, kf_id in obj_dict[f"default_{self.id}_top_kfs"]]



    def purge_points_ids(self, purge_ids: List[int]) -> None:
        """TODO: We need to define an heurisitc to purge ids that were optimized and do not fall inside the 3D instance. -> As long as we use map's point_ins_ids to classify, and not the instance saved points_ids, it doesn't matter if we don't prune.
        """
        points_ids = self.points_ids.copy()
        for point in points_ids:
            if point in purge_ids:
                self.points_ids.remove(point)

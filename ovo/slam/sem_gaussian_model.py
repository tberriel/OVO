from plyfile import PlyData, PlyElement
from pathlib import Path
from torch import nn
import numpy as np
import torch

from ..submodules.gaussian_slam.entities.gaussian_model import GaussianModel


class SemGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int = 3, isotropic=False):
        super().__init__(sh_degree, isotropic)
        self.gaussian_param_names += [
            "obj_ids"
        ]
        self._obj_ids = torch.empty(0, requires_grad=False, device="cuda", dtype=torch.int32)
        self.ids = torch.empty(0, requires_grad=False, device="cuda", dtype=torch.int32)
        self.max_id = -1

    def restore_from_params(self, params_dict, training_args):
        self.training_setup(training_args)
        self.densification_postfix(
            params_dict["xyz"],
            params_dict["features_dc"],
            params_dict["features_rest"],
            params_dict["opacity"],
            params_dict["scaling"],
            params_dict["rotation"],
            params_dict["obj_ids"],
            params_dict["ids"],
            )

    def capture_dict(self):
        super_dict = super().capture_dict()
        super_dict["obj_ids"] = self._obj_ids.clone().detach().cpu()
        super_dict["ids"] = self.ids.clone().detach().cpu()
        super_dict["max_id"] = self.max_id
        return super_dict

    def get_ids(self):
        return self.ids
    
    def get_obj_ids(self):
        return self._obj_ids.squeeze()
    
    def set_objs_ids(self, obj_ids):
        self._obj_ids = obj_ids

    def training_setup(self, training_args):
        super().training_setup(training_args)

    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        for i in range(self._obj_ids.shape[1]):
            l.append("obj_{}".format(i))
        l.append("ids")
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        obj_ids = self._obj_ids.detach().cpu().numpy()
        ids = self.ids.detach().cpu().numpy()[:,None]

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(self._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 3))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_ids, ids), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        obj_ids = np.stack((
                np.asarray(plydata.elements[0]["obj_0"]),
                np.asarray(plydata.elements[0]["obj_1"]),
                np.asarray(plydata.elements[0]["obj_2"]),
                np.asarray(plydata.elements[0]["obj_3"])),
                axis=1)
        
        ids = np.asarray(plydata.elements[0]["ids"])
        xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
                axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._obj_ids = torch.tensor(obj_ids, dtype=torch.int32, device="cuda").requires_grad_(False)
        self.ids = torch.tensor(ids, dtype=torch.int32, device="cuda").requires_grad_(False)
        self.max_id = max(ids)
        self.active_sh_degree = self.max_sh_degree

    def prune_points(self, mask):
        super().prune_points(mask)

        valid_points_mask = ~mask
        self._obj_ids = self._obj_ids[valid_points_mask]
        self.ids = self.ids[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation, new_obj_idxs = None):
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation)
        
        if new_obj_idxs is None:
            n_new_gaussians = new_xyz.shape[0]
            new_obj_idxs = torch.cat((self._obj_ids,torch.ones((n_new_gaussians,1), device="cuda",dtype=torch.int32)*(-1)), dim=0)
            new_ids = torch.tensor(list(range(self.max_id+1,self.max_id+1+n_new_gaussians)), dtype=torch.int32, device="cuda")
            
        self._obj_ids = new_obj_idxs
        self.ids = torch.cat((self.ids,new_ids))
        self.max_id = self.max_id+n_new_gaussians

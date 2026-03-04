#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import json
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._deformation = deform_network(W=args.net_width, D=args.defor_depth, 
                                           min_embeddings=args.min_embeddings, max_embeddings=args.max_embeddings, 
                                           num_frames=args.total_num_frames,
                                           args=args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._embedding = torch.empty(0)
        self._dyn_score = torch.empty(0)
        self._dyn_obs = torch.empty(0, dtype=torch.long)
        self._birth_iter = torch.empty(0, dtype=torch.long)
        self._dyn_history = None
        self._dyn_history_ptr = 0
        self._dyn_history_count = 0

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._embedding,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._dyn_score,
            self._dyn_obs,
            self._birth_iter,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) >= 17:
            (self.active_sh_degree,
            self._xyz,
            self._deformation,
            self._features_dc,
            self._features_rest,
            self._embedding,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._dyn_score,
            self._dyn_obs,
            self._birth_iter) = model_args
        else:
            (self.active_sh_degree,
            self._xyz,
            self._deformation,
            self._features_dc,
            self._features_rest,
            self._embedding,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._initialize_dynamic_states(self._xyz.shape[0], current_iter=0, device=self._xyz.device)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._dyn_score = self._dyn_score.to(self._xyz.device, dtype=torch.float32)
        self._dyn_obs = self._dyn_obs.to(self._xyz.device, dtype=torch.long)
        self._birth_iter = self._birth_iter.to(self._xyz.device, dtype=torch.long)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_deformed_features(self, dc):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_embedding(self):
        return self._embedding

    @property
    def get_dynamic_score(self):
        return self._dyn_score

    @property
    def get_dynamic_obs(self):
        return self._dyn_obs

    @property
    def get_birth_iter(self):
        return self._birth_iter
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _initialize_dynamic_states(self, num_points, current_iter=0, device=None):
        if device is None:
            device = self._xyz.device if self._xyz.numel() > 0 else "cuda"
        self._dyn_score = torch.zeros((num_points,), device=device, dtype=torch.float32)
        self._dyn_obs = torch.zeros((num_points,), device=device, dtype=torch.long)
        self._birth_iter = torch.full((num_points,), int(current_iter), device=device, dtype=torch.long)
        self._dyn_history = None
        self._dyn_history_ptr = 0
        self._dyn_history_count = 0

    def _append_dynamic_states(self, parent_indices=None, new_count=0, current_iter=0):
        if parent_indices is not None:
            parent_indices = parent_indices.long().to(self._xyz.device)
            new_count = int(parent_indices.shape[0])
        else:
            new_count = int(new_count)

        if new_count <= 0:
            return

        if parent_indices is not None and self._dyn_score.numel() > 0:
            new_score = self._dyn_score[parent_indices]
        else:
            new_score = torch.zeros((new_count,), device=self._xyz.device, dtype=torch.float32)
        new_obs = torch.zeros((new_count,), device=self._xyz.device, dtype=torch.long)
        new_birth = torch.full((new_count,), int(current_iter), device=self._xyz.device, dtype=torch.long)

        self._dyn_score = torch.cat([self._dyn_score, new_score], dim=0)
        self._dyn_obs = torch.cat([self._dyn_obs, new_obs], dim=0)
        self._birth_iter = torch.cat([self._birth_iter, new_birth], dim=0)
        self._dyn_history = None
        self._dyn_history_ptr = 0
        self._dyn_history_count = 0

    def _prune_dynamic_states(self, valid_points_mask):
        self._dyn_score = self._dyn_score[valid_points_mask]
        self._dyn_obs = self._dyn_obs[valid_points_mask]
        self._birth_iter = self._birth_iter[valid_points_mask]
        self._dyn_history = None
        self._dyn_history_ptr = 0
        self._dyn_history_count = 0

    @staticmethod
    def _percentile_normalize(values, low_percentile, high_percentile, eps=1e-6):
        low = float(np.clip(low_percentile, 0.0, 100.0)) / 100.0
        high = float(np.clip(high_percentile, 0.0, 100.0)) / 100.0
        if high < low:
            low, high = high, low

        q_low = torch.quantile(values, low)
        q_high = torch.quantile(values, high)
        denom = torch.clamp(q_high - q_low, min=eps)
        return torch.clamp((values - q_low) / denom, min=0.0, max=1.0)

    def _ensure_dyn_history(self, history_size):
        num_points = self.get_xyz.shape[0]
        if history_size <= 0 or num_points == 0:
            return
        target_shape = (history_size, num_points, 3)
        if self._dyn_history is None or self._dyn_history.shape != target_shape:
            self._dyn_history = torch.zeros(target_shape, device=self._xyz.device, dtype=torch.float32)
            self._dyn_history_ptr = 0
            self._dyn_history_count = 0

    @torch.no_grad()
    def update_dynamic_score_harmonic(self, deformed_xyz, visibility_filter,
                                      history_size=30, percentile_low=5.0, percentile_high=95.0, eps=1e-6):
        if deformed_xyz is None or visibility_filter is None or self.get_xyz.shape[0] == 0 or history_size <= 0:
            return

        self._ensure_dyn_history(int(history_size))
        if self._dyn_history is None:
            return

        visible_mask = visibility_filter.to(self._xyz.device).bool()
        if not visible_mask.any():
            return

        current_xyz = deformed_xyz.detach().to(self._xyz.device, dtype=torch.float32)
        if self._dyn_history_count == 0:
            snapshot = current_xyz.clone()
        else:
            prev_idx = (self._dyn_history_ptr - 1) % int(history_size)
            snapshot = self._dyn_history[prev_idx].clone()
            snapshot[visible_mask] = current_xyz[visible_mask]

        self._dyn_history[self._dyn_history_ptr] = snapshot
        self._dyn_history_ptr = (self._dyn_history_ptr + 1) % int(history_size)
        self._dyn_history_count = min(self._dyn_history_count + 1, int(history_size))
        self._dyn_obs[visible_mask] += 1

        if self._dyn_history_count < 2:
            return

        history = self._dyn_history[:self._dyn_history_count]
        pos_max = history.max(dim=0).values
        pos_min = history.min(dim=0).values
        displacement = torch.linalg.norm(pos_max - pos_min, dim=-1)

        pos_mean = history.mean(dim=0, keepdim=True)
        variance = ((history - pos_mean) ** 2).sum(dim=-1).mean(dim=0)

        displacement_norm = self._percentile_normalize(displacement, percentile_low, percentile_high, eps)
        variance_norm = self._percentile_normalize(variance, percentile_low, percentile_high, eps)
        inv_disp = 1.0 / (displacement_norm + eps)
        inv_var = 1.0 / (variance_norm + eps)
        self._dyn_score = 2.0 / (inv_disp + inv_var)

    @torch.no_grad()
    def update_dynamic_score_ema(self, motion_signal, visibility_filter, beta=0.02):
        if self._dyn_score.numel() == 0 or motion_signal is None or visibility_filter is None:
            return

        beta = float(np.clip(beta, 0.0, 1.0))
        visible_mask = visibility_filter.to(self._xyz.device).bool()
        if not visible_mask.any():
            return

        motion_signal = motion_signal.to(self._xyz.device, dtype=torch.float32)
        self._dyn_score[visible_mask] = (1.0 - beta) * self._dyn_score[visible_mask] + beta * motion_signal[visible_mask]
        self._dyn_obs[visible_mask] += 1

    def get_mature_mask(self, min_obs=50):
        return self._dyn_obs >= int(min_obs)

    @torch.no_grad()
    def get_dynamic_score_brief(self, min_obs=50):
        if self._dyn_score.numel() == 0:
            return {"mean": 0.0, "max": 0.0, "mature_ratio": 0.0, "num_mature": 0, "num_total": 0}
        mature_mask = self.get_mature_mask(min_obs)
        num_total = int(self._dyn_score.shape[0])
        num_mature = int(mature_mask.sum().item())
        mature_ratio = float(num_mature / max(num_total, 1))
        if num_mature == 0:
            return {"mean": 0.0, "max": 0.0, "mature_ratio": mature_ratio, "num_mature": 0, "num_total": num_total}
        mature_scores = self._dyn_score[mature_mask]
        return {
            "mean": float(mature_scores.mean().item()),
            "max": float(mature_scores.max().item()),
            "mature_ratio": mature_ratio,
            "num_mature": num_mature,
            "num_total": num_total,
        }

    @torch.no_grad()
    def export_topk_dynamic_scores(self, json_path, topk=0, min_obs=50):
        topk = int(topk)
        if topk <= 0 or self._dyn_score.numel() == 0:
            return None
        mature_mask = self.get_mature_mask(min_obs)
        mature_idx = torch.where(mature_mask)[0]
        if mature_idx.numel() == 0:
            return None
        mature_scores = self._dyn_score[mature_mask]
        k = min(topk, mature_scores.numel())
        values, indices = torch.topk(mature_scores, k=k, largest=True)
        selected_idx = mature_idx[indices]
        payload = {
            "topk": k,
            "indices": selected_idx.detach().cpu().tolist(),
            "scores": values.detach().cpu().tolist(),
            "obs": self._dyn_obs[selected_idx].detach().cpu().tolist(),
            "birth_iter": self._birth_iter[selected_idx].detach().cpu().tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return payload

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, max=1.0)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        embedding = torch.zeros((fused_color.shape[0], self._deformation.gaussian_embedding_dim)).float().cuda()  # [jm]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._embedding = nn.Parameter(embedding.requires_grad_(True))  # [jm]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._initialize_dynamic_states(self.get_xyz.shape[0], current_iter=0, device=self._xyz.device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self._deformation.offsets], 'lr': training_args.offsets_lr, "name": "offsets"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / training_args.feature_lr_div_factor, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._embedding], 'lr': training_args.feature_lr, "name": "embedding"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.deformation_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._embedding.shape[1]):
            l.append('embedding_{}'.format(i))
        return l

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        embedding = self._embedding.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, embedding), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self, ratio=0):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if ratio is not None:
            mask = torch.rand(self.get_opacity.shape[0], device="cuda") < ratio
            opacities_new[~mask] = self.get_opacity[~mask]
            print(f"reset opacity: {mask.sum()} points")
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("embedding")]
        embedding_names = sorted(embedding_names, key = lambda x: int(x.split('_')[-1]))
        embeddings = np.zeros((xyz.shape[0], len(embedding_names)))
        for idx, attr_name in enumerate(embedding_names):
            embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._embedding = nn.Parameter(torch.tensor(embeddings, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self._initialize_dynamic_states(self.get_xyz.shape[0], current_iter=0, device=self._xyz.device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "offsets":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._embedding = optimizable_tensors["embedding"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._prune_dynamic_states(valid_points_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1 or group["name"] == "offsets":continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_embedding,
                              parent_indices=None, current_iter=0):
        d = {"xyz": new_xyz, 
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,   
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "embedding" : new_embedding,
       }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._embedding = optimizable_tensors["embedding"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._append_dynamic_states(parent_indices=parent_indices, current_iter=current_iter)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, current_iter=0):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_embedding = self._embedding[selected_pts_mask].repeat(N,1)
        selected_indices = torch.where(selected_pts_mask)[0]
        parent_indices = selected_indices.repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_embedding,
                       parent_indices=parent_indices, current_iter=current_iter)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, current_iter=0):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_embedding = self._embedding[selected_pts_mask]
        parent_indices = torch.where(selected_pts_mask)[0]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_embedding,
                       parent_indices=parent_indices, current_iter=current_iter)

    def prune(self, max_grad, min_opacity, extent, max_screen_size, use_mean=False):
        if use_mean:
            prune_mask = (self.get_opacity < (self.get_opacity.max() - (self.get_opacity.max() - self.get_opacity.min())*0.5) ).squeeze()        
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, current_iter=0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, current_iter=current_iter)
        self.densify_and_split(grads, max_grad, extent, current_iter=current_iter)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)

    @torch.no_grad()
    def export_dynamic_score_stats(self, json_path, include_values=True, hist_bins=20, min_obs=50):
        if self._dyn_score.numel() == 0:
            return None

        mature_mask = self.get_mature_mask(min_obs)
        scores = self._dyn_score[mature_mask].detach().float().cpu()
        num_total = int(self._dyn_score.shape[0])
        num_mature = int(mature_mask.sum().item())
        if num_mature == 0:
            payload = {
                "num_gaussians": num_total,
                "num_mature": 0,
                "mature_ratio": 0.0,
                "min_obs": int(min_obs),
                "message": "No mature gaussians yet",
            }
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
            return payload

        scores_np = scores.numpy()
        hist_bins = max(int(hist_bins), 1)
        hist_counts, hist_edges = np.histogram(scores_np, bins=hist_bins, range=(0.0, 1.0))

        payload = {
            "num_gaussians": num_total,
            "num_mature": num_mature,
            "mature_ratio": float(num_mature / max(num_total, 1)),
            "min_obs": int(min_obs),
            "min": float(scores_np.min()),
            "max": float(scores_np.max()),
            "mean": float(scores_np.mean()),
            "std": float(scores_np.std()),
            "p01": float(np.percentile(scores_np, 1)),
            "p05": float(np.percentile(scores_np, 5)),
            "p10": float(np.percentile(scores_np, 10)),
            "p25": float(np.percentile(scores_np, 25)),
            "p50": float(np.percentile(scores_np, 50)),
            "p75": float(np.percentile(scores_np, 75)),
            "p90": float(np.percentile(scores_np, 90)),
            "p95": float(np.percentile(scores_np, 95)),
            "p99": float(np.percentile(scores_np, 99)),
            "histogram": {
                "bins": hist_bins,
                "range": [0.0, 1.0],
                "counts": hist_counts.tolist(),
                "edges": hist_edges.tolist(),
            },
        }

        if include_values:
            payload["scores"] = scores_np.tolist()

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return payload

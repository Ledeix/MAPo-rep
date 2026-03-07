import copy
import json
import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SegmentRecord:
    seg_id: int
    t_start: float
    t_end: float
    level: int
    parent_seg_id: int
    active: bool = True


class SegmentManager(nn.Module):
    def __init__(self, args, base_deformation, total_time: float):
        super().__init__()
        self.enabled = bool(getattr(args, "enable_temporal_partition", False))
        self.total_time = float(total_time)

        self.max_level = int(getattr(args, "temporal_max_level", 4))
        self.max_segments_total = int(getattr(args, "temporal_max_segments", 32))
        self.min_obs = int(getattr(args, "dynamic_score_min_obs", 50))
        self.min_candidates = int(getattr(args, "temporal_min_candidates", 1000))
        self.min_split_count = int(getattr(args, "temporal_min_split_count", 100))
        self.one_split_per_interval = bool(getattr(args, "temporal_one_split_per_interval", True))

        self.quantiles_by_level = {
            0: float(getattr(args, "temporal_q_level0", 0.90)),
            1: float(getattr(args, "temporal_q_level1", 0.95)),
            2: float(getattr(args, "temporal_q_level2", 0.98)),
            3: float(getattr(args, "temporal_q_level3", 0.99)),
        }

        self.records = {0: SegmentRecord(seg_id=0, t_start=0.0, t_end=self.total_time, level=0, parent_seg_id=-1, active=True)}
        self.next_seg_id = 1
        self.tau_by_level = {}
        self.last_cross_stats = {
            "iteration": -1,
            "num_views": 0,
            "lself": 0.0,
            "lgt": 0.0,
            "total": 0.0,
        }

        self.deform_nets = nn.ModuleDict({"0": base_deformation})

    def _to_time_scalar(self, t):
        time_scalar = float(t)
        if self.total_time > 1.5 and time_scalar <= 1.0 + 1e-6:
            time_scalar = time_scalar * self.total_time
        return time_scalar

    def get_net(self, seg_id: int):
        return self.deform_nets[str(int(seg_id))]

    def num_segments_total(self):
        return len(self.records)

    def num_segments_active(self):
        return sum(1 for s in self.records.values() if s.active)

    def _create_child(self, parent_id: int, t_start: float, t_end: float, level: int):
        seg_id = self.next_seg_id
        self.next_seg_id += 1
        self.records[seg_id] = SegmentRecord(
            seg_id=seg_id,
            t_start=float(t_start),
            t_end=float(t_end),
            level=int(level),
            parent_seg_id=int(parent_id),
            active=True,
        )
        parent_net = self.get_net(parent_id)
        self.deform_nets[str(seg_id)] = copy.deepcopy(parent_net)
        return seg_id

    def ensure_nets_for_records(self, records):
        # Recreate missing per-segment deformation nets before loading a state dict.
        if records is None:
            return
        for seg_id in sorted([int(k) for k in records.keys()]):
            key = str(seg_id)
            if key in self.deform_nets:
                continue
            rec = records[seg_id]
            parent_id = int(rec.parent_seg_id)
            parent_key = str(parent_id) if parent_id >= 0 and str(parent_id) in self.deform_nets else "0"
            self.deform_nets[key] = copy.deepcopy(self.deform_nets[parent_key])

    def deform_subset(self, pc, idx_tensor, time_values, cam_no=None, iter_idx=0, num_down_emb_c=30, num_down_emb_f=30):
        if idx_tensor.numel() == 0:
            empty_xyz = torch.empty((0, 3), device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
            empty_scale = torch.empty((0, pc._scaling.shape[1]), device=pc.get_xyz.device, dtype=pc._scaling.dtype)
            empty_rot = torch.empty((0, pc._rotation.shape[1]), device=pc.get_xyz.device, dtype=pc._rotation.dtype)
            empty_opa = torch.empty((0, pc._opacity.shape[1]), device=pc.get_xyz.device, dtype=pc._opacity.dtype)
            empty_shs = torch.empty((0, pc.get_features.shape[1], pc.get_features.shape[2]), device=pc.get_xyz.device, dtype=pc.get_features.dtype)
            return empty_xyz, empty_scale, empty_rot, empty_opa, empty_shs, None

        idx_tensor = idx_tensor.long().to(pc.get_xyz.device)
        if time_values.ndim == 1:
            time_values = time_values.unsqueeze(-1)
        time_values = time_values.to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)

        means = pc._xyz[idx_tensor]
        scales = pc._scaling[idx_tensor]
        rotations = pc._rotation[idx_tensor]
        opacity = pc._opacity[idx_tensor]
        shs = pc.get_features[idx_tensor]
        seg_ids = pc._seg_id_g[idx_tensor]

        means_out = means.clone()
        scales_out = scales.clone()
        rotations_out = rotations.clone()
        opacity_out = opacity.clone()
        shs_out = shs.clone()
        extras = None

        for seg_id_tensor in torch.unique(seg_ids):
            seg_id = int(seg_id_tensor.item())
            local_idx = torch.where(seg_ids == seg_id)[0]
            if local_idx.numel() == 0:
                continue
            global_idx = idx_tensor[local_idx]
            net = self.get_net(seg_id)
            out = net(
                means[local_idx],
                scales=scales[local_idx],
                rotations=rotations[local_idx],
                opacity=opacity[local_idx],
                time_emb=time_values[local_idx],
                cam_no=cam_no,
                pc=None,
                embeddings=pc.get_embedding[global_idx],
                sh_coefs=shs[local_idx],
                iter=iter_idx,
                num_down_emb_c=num_down_emb_c,
                num_down_emb_f=num_down_emb_f,
            )
            means_out[local_idx] = out[0]
            scales_out[local_idx] = out[1]
            rotations_out[local_idx] = out[2]
            opacity_out[local_idx] = out[3]
            shs_out[local_idx] = out[4]
            extras = out[5]

        return means_out, scales_out, rotations_out, opacity_out, shs_out, extras

    def forward_deformation(self, pc, means3D, scales, rotations, opacity, time, cam_no, shs, iter_idx,
                            num_down_emb_c=30, num_down_emb_f=30, force_segment_id=None):
        if means3D.shape[0] == 0:
            empty_mask = torch.zeros((0,), device=means3D.device, dtype=torch.bool)
            out = (means3D, scales, rotations, opacity, shs, None)
            return out, empty_mask

        if time.numel() == 0:
            time_scalar = 0.0
        else:
            time_scalar = self._to_time_scalar(time.reshape(-1)[0].item())

        if not self.enabled:
            return pc._deformation(
                means3D, scales, rotations, opacity, time, cam_no, pc, None, shs,
                iter=iter_idx, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f,
            ), torch.ones((means3D.shape[0],), device=means3D.device, dtype=torch.bool)

        if force_segment_id is None:
            active_mask = (time_scalar >= pc._t_start_g) & (time_scalar < pc._t_end_g)
        else:
            active_mask = (pc._seg_id_g == int(force_segment_id)) & (time_scalar >= pc._t_start_g) & (time_scalar < pc._t_end_g)

        means3D_final = means3D.clone()
        scales_final = scales.clone()
        rotations_final = rotations.clone()
        opacity_final = opacity.clone()
        shs_final = shs.clone()
        extras = None

        freeze_static = bool(getattr(pc._training_args, "static_freeze_deform_for_static", True)) if pc._training_args is not None else True
        active_static = active_mask & pc._is_static_g if freeze_static else torch.zeros_like(active_mask)
        active_dynamic = active_mask & (~active_static)

        if active_static.any():
            means3D_final[active_static] = pc._xyz_static[active_static]
            scales_final[active_static] = pc._scaling_static[active_static]
            rotations_final[active_static] = pc._rotation_static[active_static]
            opacity_final[active_static] = pc._opacity_static[active_static]
            shs_final[active_static] = pc.get_static_features[active_static]

        if active_dynamic.any():
            if force_segment_id is None:
                seg_ids = torch.unique(pc._seg_id_g[active_dynamic])
            else:
                seg_ids = torch.tensor([int(force_segment_id)], device=pc._seg_id_g.device, dtype=pc._seg_id_g.dtype)
            for seg_id_tensor in seg_ids:
                seg_id = int(seg_id_tensor.item())
                idx = torch.where(active_dynamic & (pc._seg_id_g == seg_id))[0]
                if idx.numel() == 0:
                    continue
                net = self.get_net(seg_id)
                out = net(
                    means3D[idx],
                    scales=scales[idx],
                    rotations=rotations[idx],
                    opacity=opacity[idx],
                    time_emb=time[idx],
                    cam_no=cam_no,
                    pc=None,
                    embeddings=pc.get_embedding[idx],
                    sh_coefs=shs[idx],
                    iter=iter_idx,
                    num_down_emb_c=num_down_emb_c,
                    num_down_emb_f=num_down_emb_f,
                )
                means3D_final[idx] = out[0]
                scales_final[idx] = out[1]
                rotations_final[idx] = out[2]
                opacity_final[idx] = out[3]
                shs_final[idx] = out[4]
                extras = out[5]

        inactive_mask = ~active_mask
        if inactive_mask.any():
            opacity_final[inactive_mask] = -100.0

        return (means3D_final, scales_final, rotations_final, opacity_final, shs_final, extras), active_mask

    def get_segment_for_time(self, t):
        t = self._to_time_scalar(t)
        covering = [rec for rec in self.records.values() if rec.active and rec.t_start <= t < rec.t_end]
        if len(covering) == 0:
            return None
        covering.sort(key=lambda r: (r.level, r.t_end - r.t_start), reverse=True)
        return int(covering[0].seg_id)

    def get_boundary_distance(self, seg_id, t):
        rec = self.records.get(int(seg_id), None)
        if rec is None:
            return float("inf")
        t = self._to_time_scalar(t)
        return min(abs(t - rec.t_start), abs(t - rec.t_end))

    def get_neighbor_segment(self, seg_id, t):
        seg_id = int(seg_id)
        rec = self.records.get(seg_id, None)
        if rec is None:
            return None

        siblings = [
            s for s in self.records.values()
            if s.active and s.seg_id != seg_id and s.parent_seg_id == rec.parent_seg_id and s.level == rec.level
        ]
        if len(siblings) > 0:
            siblings.sort(key=lambda s: min(abs(rec.t_start - s.t_end), abs(rec.t_end - s.t_start)))
            return int(siblings[0].seg_id)

        candidates = [s for s in self.records.values() if s.active and s.seg_id != seg_id]
        if len(candidates) == 0:
            return None

        t = self._to_time_scalar(t)

        def _score(s):
            edge_gap = min(abs(rec.t_start - s.t_end), abs(rec.t_end - s.t_start))
            boundary_dist = min(abs(t - s.t_start), abs(t - s.t_end))
            return (edge_gap, boundary_dist)

        candidates.sort(key=_score)
        return int(candidates[0].seg_id)

    def update_cross_stats(self, iteration, num_views, lself, lgt, total):
        self.last_cross_stats = {
            "iteration": int(iteration),
            "num_views": int(num_views),
            "lself": float(lself),
            "lgt": float(lgt),
            "total": float(total),
        }

    @torch.no_grad()
    def update_taus(self, gaussians):
        dyn_score = gaussians._dyn_score
        dyn_obs = gaussians._dyn_obs
        level_g = gaussians._level_g

        for level in range(self.max_level):
            q = self.quantiles_by_level.get(level, self.quantiles_by_level[max(self.quantiles_by_level.keys())])
            candidates = (level_g == level) & (dyn_obs >= self.min_obs)
            num_candidates = int(candidates.sum().item())
            if num_candidates < self.min_candidates:
                continue
            self.tau_by_level[level] = float(torch.quantile(dyn_score[candidates], q).item())

    @torch.no_grad()
    def _candidate_stats(self, gaussians, seg_id: int, level: int):
        tau = self.tau_by_level.get(level, None)
        if tau is None:
            return None, 0
        seg_mask = gaussians._seg_id_g == seg_id
        mask = seg_mask & (gaussians._level_g == level) & (gaussians._dyn_obs >= self.min_obs) & (gaussians._dyn_score > tau)
        count = int(mask.sum().item())
        if count == 0:
            return None, 0
        score = float(gaussians._dyn_score[mask].mean().item())
        return mask, score

    @torch.no_grad()
    def maybe_split(self, gaussians, current_iter: int):
        if not self.enabled:
            return []
        if self.num_segments_total() >= self.max_segments_total:
            return []

        split_candidates = []
        for seg_id, rec in self.records.items():
            if (not rec.active) or rec.level >= self.max_level:
                continue
            mask, score = self._candidate_stats(gaussians, seg_id, rec.level)
            if mask is None:
                continue
            count = int(mask.sum().item())
            if count < self.min_split_count:
                continue
            split_candidates.append((score, seg_id, mask, count))

        if len(split_candidates) == 0:
            return []

        split_candidates.sort(key=lambda x: x[0], reverse=True)
        picked = split_candidates[:1] if self.one_split_per_interval else split_candidates

        events = []
        for score, seg_id, mask, count in picked:
            if self.num_segments_total() + 2 > self.max_segments_total:
                break
            ev = self.split_segment(gaussians, seg_id, mask, current_iter)
            ev["mean_dyn_score"] = score
            ev["split_count"] = count
            events.append(ev)
        return events

    @torch.no_grad()
    def split_segment(self, gaussians, parent_seg_id: int, split_mask, current_iter: int):
        parent = self.records[int(parent_seg_id)]
        a, b = parent.t_start, parent.t_end
        mid = 0.5 * (a + b)
        next_level = parent.level + 1

        seg_l = self._create_child(parent_seg_id, a, mid, next_level)
        seg_r = self._create_child(parent_seg_id, mid, b, next_level)

        gaussians.register_new_segment_optimizer_params([seg_l, seg_r])

        idx = torch.where(split_mask)[0]
        old_n = gaussians.get_xyz.shape[0]
        new_idx = gaussians.append_gaussian_clones(idx, current_iter=current_iter)

        gaussians._seg_id_g[idx] = int(seg_l)
        gaussians._t_start_g[idx] = float(a)
        gaussians._t_end_g[idx] = float(mid)
        gaussians._level_g[idx] = int(next_level)
        gaussians._dyn_obs[idx] = 0
        gaussians._birth_iter[idx] = int(current_iter)

        gaussians._seg_id_g[new_idx] = int(seg_r)
        gaussians._t_start_g[new_idx] = float(mid)
        gaussians._t_end_g[new_idx] = float(b)
        gaussians._level_g[new_idx] = int(next_level)
        gaussians._dyn_obs[new_idx] = 0
        gaussians._birth_iter[new_idx] = int(current_iter)

        gaussians._is_static_g[idx] = False
        gaussians._xyz_static[idx] = gaussians._xyz[idx]
        gaussians._scaling_static[idx] = gaussians._scaling[idx]
        gaussians._rotation_static[idx] = gaussians._rotation[idx]
        gaussians._opacity_static[idx] = gaussians._opacity[idx]
        gaussians._features_dc_static[idx] = gaussians._features_dc[idx]
        gaussians._features_rest_static[idx] = gaussians._features_rest[idx]

        gaussians._is_static_g[new_idx] = False
        gaussians._xyz_static[new_idx] = gaussians._xyz[new_idx]
        gaussians._scaling_static[new_idx] = gaussians._scaling[new_idx]
        gaussians._rotation_static[new_idx] = gaussians._rotation[new_idx]
        gaussians._opacity_static[new_idx] = gaussians._opacity[new_idx]
        gaussians._features_dc_static[new_idx] = gaussians._features_dc[new_idx]
        gaussians._features_rest_static[new_idx] = gaussians._features_rest[new_idx]

        gaussians.assert_metadata_alignment()

        return {
            "parent_seg_id": int(parent_seg_id),
            "left_seg_id": int(seg_l),
            "right_seg_id": int(seg_r),
            "range": [float(a), float(b)],
            "mid": float(mid),
            "level": int(parent.level),
            "new_level": int(next_level),
            "num_selected": int(idx.numel()),
            "old_num_gaussians": int(old_n),
            "new_num_gaussians": int(gaussians.get_xyz.shape[0]),
        }

    @torch.no_grad()
    def debug_dump(self, gaussians, model_path: str, iteration: int, settings_dict=None):
        if not self.enabled:
            return None
        debug_dir = os.path.join(model_path, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        segment_entries = []
        for seg_id, rec in sorted(self.records.items(), key=lambda x: x[0]):
            seg_mask = gaussians._seg_id_g == int(seg_id)
            mature_mask = seg_mask & (gaussians._dyn_obs >= self.min_obs)
            num_seg = int(seg_mask.sum().item())
            num_mature = int(mature_mask.sum().item())
            if num_mature > 0:
                mean_dyn = float(gaussians._dyn_score[mature_mask].mean().item())
                max_dyn = float(gaussians._dyn_score[mature_mask].max().item())
            else:
                mean_dyn, max_dyn = 0.0, 0.0
            segment_entries.append({
                "seg_id": int(seg_id),
                "level": int(rec.level),
                "t_start": float(rec.t_start),
                "t_end": float(rec.t_end),
                "parent_seg_id": int(rec.parent_seg_id),
                "active": bool(rec.active),
                "num_gaussians": num_seg,
                "num_mature": num_mature,
                "mean_dyn_score_mature": mean_dyn,
                "max_dyn_score_mature": max_dyn,
            })

        by_level = {}
        for level in range(self.max_level + 1):
            level_mask = gaussians._level_g == level
            cand_mask = level_mask & (gaussians._dyn_obs >= self.min_obs)
            tau = self.tau_by_level.get(level, None)
            if tau is not None:
                split_mask = cand_mask & (gaussians._dyn_score > tau)
                num_split = int(split_mask.sum().item())
            else:
                num_split = 0
            by_level[str(level)] = {
                "tau": None if tau is None else float(tau),
                "num_candidates": int(cand_mask.sum().item()),
                "num_to_split": num_split,
            }

        segment_entries_sorted = sorted(segment_entries, key=lambda x: x["mean_dyn_score_mature"], reverse=True)
        top5 = segment_entries_sorted[:5]
        print(f"[ITER {iteration}] segments total={self.num_segments_total()} active={self.num_segments_active()}")
        for row in top5:
            print(
                f"  seg {row['seg_id']} lvl={row['level']} range=[{row['t_start']:.3f},{row['t_end']:.3f}) "
                f"n={row['num_gaussians']} mature={row['num_mature']} mean={row['mean_dyn_score_mature']:.6f} max={row['max_dyn_score_mature']:.6f}"
            )

        payload = {
            "iteration": int(iteration),
            "segments": segment_entries,
            "tau_by_level": {str(k): float(v) for k, v in self.tau_by_level.items()},
            "level_stats": by_level,
            "settings": settings_dict or {},
            "cross_loss": self.last_cross_stats,
        }

        out_path = os.path.join(debug_dir, f"segments_iter{iteration:06d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

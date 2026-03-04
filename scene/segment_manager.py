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

        self.deform_nets = nn.ModuleDict({"0": base_deformation})

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

    def forward_deformation(self, pc, means3D, scales, rotations, opacity, time, cam_no, shs, iter_idx, num_down_emb_c=30, num_down_emb_f=30):
        if means3D.shape[0] == 0:
            empty_mask = torch.zeros((0,), device=means3D.device, dtype=torch.bool)
            out = (means3D, scales, rotations, opacity, shs, None)
            return out, empty_mask

        if time.numel() == 0:
            time_scalar = 0.0
        else:
            time_scalar = float(time.reshape(-1)[0].item())

        if self.total_time > 1.5 and time_scalar <= 1.0 + 1e-6:
            time_scalar = time_scalar * self.total_time

        if not self.enabled:
            return pc._deformation(
                means3D, scales, rotations, opacity, time, cam_no, pc, None, shs,
                iter=iter_idx, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f,
            ), torch.ones((means3D.shape[0],), device=means3D.device, dtype=torch.bool)

        active_mask = (time_scalar >= pc._t_start_g) & (time_scalar < pc._t_end_g)

        means3D_final = means3D.clone()
        scales_final = scales.clone()
        rotations_final = rotations.clone()
        opacity_final = opacity.clone()
        shs_final = shs.clone()
        extras = None

        if active_mask.any():
            seg_ids = torch.unique(pc._seg_id_g[active_mask])
            for seg_id_tensor in seg_ids:
                seg_id = int(seg_id_tensor.item())
                idx = torch.where(active_mask & (pc._seg_id_g == seg_id))[0]
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
        }

        out_path = os.path.join(debug_dir, f"segments_iter{iteration:06d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

# Main evaluator implementation and backend orchestration.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

from .backends import ColmapBackend, VGGTBackend
from .config import CONFIG
from .metrics import ObjectMetricsEngine
from .utils import chamfer_and_fscore, cosine_similarity, image_metrics, parse_frame_idx, rotation_error_deg
from .visualization import create_combined_visualization


class ScanNetEvaluator:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        if device.startswith("cuda") and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.scene_cache: Dict[str, Dict[str, Any]] = {}

        self.colmap_backend = ColmapBackend()
        self.vggt_backend = VGGTBackend(model=self.model, device=self.device, dtype=self.dtype)
        self.object_metrics_engine = ObjectMetricsEngine(device=self.device)

        # Kept for CLI compatibility checks.
        self.colmap_bin = self.colmap_backend.colmap_bin

    def load_scene_meta(self, scene_id: str) -> Optional[Dict[str, Any]]:
        if scene_id in self.scene_cache:
            return self.scene_cache[scene_id]

        pkl_path = os.path.join(CONFIG["INFO_DIR"], f"{scene_id}.pkl")
        if not os.path.exists(pkl_path):
            return None

        try:
            import pickle

            with open(pkl_path, "rb") as f:
                pack = pickle.load(f)

            datalist = pack["data_list"]
            target_entry = datalist[0]
            if len(datalist) > 1:
                for entry in datalist:
                    if entry.get("images") and scene_id in entry["images"][0]["img_path"]:
                        target_entry = entry
                        break

            axis_align = target_entry["axis_align_matrix"]
            frame_dict = {}
            for img_info in target_entry["images"]:
                fname = os.path.basename(img_info["img_path"])
                p = axis_align @ img_info["cam2global"]
                frame_dict[fname] = {
                    "pose": p,
                    "raw_path": img_info["img_path"],
                    "visible_instance_ids": set(img_info.get("visible_instance_ids", [])),
                }

            categories = pack.get("metainfo", {}).get("categories", {})
            id2name = {int(v): k for k, v in categories.items()}

            instances = {}
            for inst in target_entry.get("instances", []):
                bbox_id = int(inst.get("bbox_id", -1))
                bbox = inst.get("bbox_3d", [0.0] * 9)
                if len(bbox) < 7:
                    continue
                label_id = int(inst.get("bbox_label_3d", -1))
                center = np.array(bbox[:3], dtype=np.float64)
                size = np.array(bbox[3:6], dtype=np.float64)
                yaw = float(bbox[6])
                instances[bbox_id] = {
                    "center": center,
                    "size": size,
                    "yaw": yaw,
                    "label_id": label_id,
                    "label_name": id2name.get(label_id, str(label_id)),
                }

            cache_item = {
                "frames": frame_dict,
                "instances": instances,
                "cam2img": np.asarray(target_entry.get("cam2img", np.eye(4)), dtype=np.float64),
            }
            self.scene_cache[scene_id] = cache_item
            return cache_item
        except Exception:
            return None

    def resolve_abs_path(self, raw_path: str) -> str:
        if raw_path.startswith("scannet") or raw_path.startswith("/scannet"):
            clean_raw = raw_path.lstrip("/").replace("scannet/", "", 1)
            return os.path.join(CONFIG["IMAGE_ROOT"], clean_raw)
        return os.path.join(CONFIG["IMAGE_ROOT"], raw_path)

    def resolve_frame(self, meta: Dict[str, Any], rel_or_name: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Set[int]]]:
        fname = os.path.basename(rel_or_name)
        frame = meta["frames"].get(fname)
        if frame is None:
            return None, None, None
        abs_path = self.resolve_abs_path(frame["raw_path"])
        return abs_path, frame["pose"], frame.get("visible_instance_ids", set())

    def get_umeyama_transform(self, P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        n = min(len(P), len(Q))
        P = P[:n]
        Q = Q[:n]
        P_m, Q_m = P.mean(0), Q.mean(0)
        P_n, Q_n = P - P_m, Q - Q_m
        C = np.dot(Q_n.T, P_n) / n
        U, S, Vh = np.linalg.svd(C)
        R = U @ Vh
        if np.linalg.det(R) < 0:
            Vh[2, :] *= -1
            R = U @ Vh
        var_p = np.var(P_n, axis=0).sum()
        s = np.trace(np.diag(S)) / var_p if var_p > 1e-8 else 1.0
        t = Q_m - s * np.dot(R, P_m)
        return float(s), R, t

    def estimate_scale_from_steps(self, pred_ctx_centers: np.ndarray, gt_ctx_centers: np.ndarray) -> float:
        if pred_ctx_centers.shape[0] < 2 or gt_ctx_centers.shape[0] < 2:
            return 1.0
        pred_steps = np.linalg.norm(np.diff(pred_ctx_centers, axis=0), axis=1)
        gt_steps = np.linalg.norm(np.diff(gt_ctx_centers, axis=0), axis=1)
        valid = pred_steps > 1e-6
        if valid.sum() == 0:
            return 1.0
        ratios = gt_steps[valid] / pred_steps[valid]
        return float(np.median(ratios))

    def align_pose_with_context_prior(
        self,
        pred_c2ws: np.ndarray,
        gt_context_poses: np.ndarray,
    ) -> Dict[str, Any]:
        n_context = len(gt_context_poses)
        pred_ctx = pred_c2ws[:n_context]

        pred_centers_ctx = pred_ctx[:, :3, 3]
        gt_centers_ctx = gt_context_poses[:, :3, 3]

        s, R_align, t_align = self.get_umeyama_transform(pred_centers_ctx, gt_centers_ctx)

        pred_target_pose = pred_c2ws[-1].copy()
        pos_align = s * (R_align @ pred_target_pose[:3, 3]) + t_align
        R_target_align = R_align @ pred_target_pose[:3, :3]

        gt_last = gt_context_poses[-1]
        pred_last = pred_ctx[-1]
        rel_pred = np.linalg.inv(pred_last) @ pred_target_pose
        scale_rel = self.estimate_scale_from_steps(pred_centers_ctx, gt_centers_ctx)

        rel_t = rel_pred[:3, 3] * scale_rel
        pos_rel = gt_last[:3, 3] + gt_last[:3, :3] @ rel_t
        R_rel = gt_last[:3, :3] @ rel_pred[:3, :3]

        alpha = float(CONFIG["POSE_PRIOR_BLEND"])
        alpha = max(0.0, min(1.0, alpha))
        fused_pos = alpha * pos_rel + (1.0 - alpha) * pos_align
        fused_R = R_rel

        fused_pose = np.eye(4, dtype=np.float64)
        fused_pose[:3, :3] = fused_R
        fused_pose[:3, 3] = fused_pos

        aligned_pred_centers_ctx = (s * np.dot(pred_centers_ctx, R_align.T)) + t_align
        ctx_rmse = float(np.sqrt(np.mean(np.sum((aligned_pred_centers_ctx - gt_centers_ctx) ** 2, axis=1))))

        return {
            "fused_pose": fused_pose,
            "sim_aligned_pos": pos_align,
            "sim_aligned_R": R_target_align,
            "aligned_pred_centers_ctx": aligned_pred_centers_ctx,
            "ctx_rmse": ctx_rmse,
            "ctx_scale": s,
        }

    def compute_pose_metrics(
        self,
        fused_pose: np.ndarray,
        gt_target_pose: np.ndarray,
        gt_last_pose: np.ndarray,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        t_err = np.linalg.norm(fused_pose[:3, 3] - gt_target_pose[:3, 3])
        r_err = rotation_error_deg(fused_pose[:3, :3], gt_target_pose[:3, :3])

        rel_gt = np.linalg.inv(gt_last_pose) @ gt_target_pose
        rel_pred = np.linalg.inv(gt_last_pose) @ fused_pose

        rpe_trans = np.linalg.norm(rel_pred[:3, 3] - rel_gt[:3, 3])
        rpe_dir = cosine_similarity(rel_pred[:3, 3], rel_gt[:3, 3])
        rpe_rot = rotation_error_deg(rel_pred[:3, :3], rel_gt[:3, :3])

        out["pose_t_err_m"] = float(t_err)
        out["pose_r_err_deg"] = float(r_err)
        out["rpe_trans_err_m"] = float(rpe_trans)
        out["rpe_trans_dir_cos"] = float(rpe_dir)
        out["rpe_rot_deg"] = float(rpe_rot)
        return out

    def run_colmap_sfm_inference(
        self,
        image_paths: List[str],
        target_abs_path: str,
        cam2img: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.colmap_backend.run_inference(image_paths=image_paths, target_abs_path=target_abs_path, cam2img=cam2img)

    def run_vggt_inference(self, image_paths: List[str], need_points: bool = False) -> Optional[Dict[str, Any]]:
        return self.vggt_backend.run_inference(image_paths=image_paths, need_points=need_points)

    def build_context_frames(self, meta: Dict[str, Any], item: Dict[str, Any]) -> Optional[Tuple[List[str], List[np.ndarray]]]:
        all_frames = []
        for fname in meta["frames"].keys():
            idx = parse_frame_idx(fname)
            if idx >= 0:
                all_frames.append({"name": fname, "idx": idx})
        all_frames.sort(key=lambda x: x["idx"])

        if not item.get("context"):
            return None

        given_first_name = os.path.basename(item["context"][0])
        given_last_name = os.path.basename(item["context"][-1])

        idx_first = -1
        idx_last = -1
        for i, f in enumerate(all_frames):
            if f["name"] == given_first_name:
                idx_first = i
            if f["name"] == given_last_name:
                idx_last = i

        if idx_last == -1:
            return None
        if idx_first == -1:
            idx_first = idx_last

        target_len = 40
        slice_end = idx_last + 1
        slice_start = max(0, slice_end - target_len)
        if slice_start > idx_first:
            slice_start = max(0, slice_end - target_len)

        selected_frames = all_frames[slice_start:slice_end]
        c_paths = []
        c_poses = []
        for f in selected_frames:
            abs_path, pose, _ = self.resolve_frame(meta, f["name"])
            if abs_path and pose is not None and os.path.exists(abs_path):
                c_paths.append(abs_path)
                c_poses.append(pose)

        if len(c_paths) < CONFIG["POSE_PRIOR_MIN_CTX"]:
            return None
        return c_paths, c_poses

    def evaluate_single_item(self, item: Dict[str, Any], idx: int, do_vis: bool = False, vis_save_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        target_rel = item.get("target", "")
        parts = target_rel.split("/")
        if len(parts) < 2:
            return None
        scene_id = parts[1]

        meta = self.load_scene_meta(scene_id)
        if not meta:
            return None

        gt_path, gt_pose, gt_visible_ids = self.resolve_frame(meta, target_rel)
        if not gt_path or gt_pose is None or not os.path.exists(gt_path):
            return None

        built = self.build_context_frames(meta, item)
        if built is None:
            return None
        c_paths, c_poses = built

        pred_img_path = Path(item.get("pred", ""))
        if not pred_img_path.exists():
            pred_img_path = Path(CONFIG["INPUT_DIR"]) / item.get("pred", "")
        if not pred_img_path.exists():
            return None

        inference_paths_pred = c_paths + [str(pred_img_path)]

        pose_backend_req = str(CONFIG.get("POSE_BACKEND", "auto")).lower()
        inf_pred = None
        if pose_backend_req in {"auto", "colmap"}:
            inf_pred = self.run_colmap_sfm_inference(
                image_paths=inference_paths_pred,
                target_abs_path=str(pred_img_path),
                cam2img=meta.get("cam2img"),
            )
        if inf_pred is None and pose_backend_req in {"auto", "vggt"}:
            inf_pred = self.run_vggt_inference(inference_paths_pred, need_points=CONFIG["ENABLE_POINT_METRICS"])
        if inf_pred is None:
            return None

        path_to_pose: Dict[str, np.ndarray] = inf_pred.get("path_to_pose", {})
        matched_gt_context = []
        matched_pred_context = []
        for cp, gp in zip(c_paths, c_poses):
            pp = path_to_pose.get(cp)
            if pp is None:
                continue
            matched_gt_context.append(gp)
            matched_pred_context.append(pp)

        target_pred_pose = path_to_pose.get(str(pred_img_path))
        if target_pred_pose is None:
            c2ws_arr = inf_pred.get("c2ws")
            if isinstance(c2ws_arr, np.ndarray) and c2ws_arr.shape[0] == len(inference_paths_pred):
                target_pred_pose = c2ws_arr[-1]
                matched_gt_context = c_poses
                matched_pred_context = list(c2ws_arr[: len(c_poses)])
            else:
                return None

        min_ctx = int(CONFIG["POSE_PRIOR_MIN_CTX"])
        if inf_pred.get("backend_used") == "colmap":
            min_ctx = int(CONFIG.get("COLMAP_MIN_MATCHED_CONTEXT", min_ctx))
        if len(matched_gt_context) < min_ctx:
            return None

        gt_context_poses = np.asarray(matched_gt_context)
        pred_c2ws = np.asarray(matched_pred_context + [target_pred_pose])

        fused = self.align_pose_with_context_prior(pred_c2ws, gt_context_poses)
        fused_pose = fused["fused_pose"]

        pose_metrics = self.compute_pose_metrics(fused_pose, gt_pose, gt_context_poses[-1])
        pose_metrics["ctx_align_rmse_m"] = float(fused["ctx_rmse"])
        pose_metrics["ctx_scale"] = float(fused["ctx_scale"])

        result: Dict[str, Any] = {
            "index": float(idx),
            "n_context": float(len(gt_context_poses)),
            "pose_backend": inf_pred.get("backend_used", "unknown"),
        }
        result.update(pose_metrics)

        if CONFIG["ENABLE_PIXEL_METRICS"]:
            result.update(image_metrics(str(pred_img_path), gt_path))

        if CONFIG["ENABLE_POINT_METRICS"]:
            pred_target_pc = inf_pred.get("target_points_world", np.zeros((0, 3), dtype=np.float32))

            inf_gt = None
            if inf_pred.get("backend_used") == "colmap":
                inf_gt = self.run_colmap_sfm_inference(
                    image_paths=c_paths + [gt_path],
                    target_abs_path=gt_path,
                    cam2img=meta.get("cam2img"),
                )
            if inf_gt is None:
                inf_gt = self.run_vggt_inference(c_paths + [gt_path], need_points=True)

            if inf_gt is not None:
                gt_target_pc = inf_gt.get("target_points_world", np.zeros((0, 3), dtype=np.float32))

                if (gt_target_pc is None or len(gt_target_pc) == 0) and inf_gt.get("pointclouds_world") is not None:
                    gt_pcs = inf_gt.get("pointclouds_world")
                    if gt_pcs is not None and len(gt_pcs) > 0:
                        gt_target_pc = gt_pcs[-1]
                if (pred_target_pc is None or len(pred_target_pc) == 0) and inf_pred.get("pointclouds_world") is not None:
                    pred_pcs = inf_pred.get("pointclouds_world")
                    if pred_pcs is not None and len(pred_pcs) > 0:
                        pred_target_pc = pred_pcs[-1]

                if pred_target_pc is not None and gt_target_pc is not None:
                    pc_metrics = chamfer_and_fscore(
                        pred_target_pc,
                        gt_target_pc,
                        f_thresh=float(CONFIG["POINT_F_THRESH"]),
                        max_samples=int(CONFIG["POINT_MAX_SAMPLES"]),
                        seed=int(CONFIG["RANDOM_SEED"]) + idx,
                    )
                    result.update(pc_metrics)
                else:
                    result.update({"pc_chamfer": float("nan"), "pc_fscore": float("nan")})
            else:
                result.update({"pc_chamfer": float("nan"), "pc_fscore": float("nan")})

        if CONFIG["ENABLE_OBJECT_METRICS"]:
            obj_source = str(CONFIG.get("OBJECT_SOURCE", "auto")).lower()
            has_anno = bool(meta.get("instances"))

            try:
                gt_img = Image.open(gt_path)
                image_wh = gt_img.size
            except Exception:
                image_wh = (640, 480)

            obj_metrics: Dict[str, float]
            if obj_source in {"annotation", "auto"} and has_anno:
                obj_metrics = self.object_metrics_engine.compute_from_annotations(
                    instances=meta.get("instances", {}),
                    gt_visible_ids=gt_visible_ids or set(),
                    gt_pose=gt_pose,
                    pred_pose=fused_pose,
                    K=meta.get("cam2img", np.eye(4)),
                    image_wh=image_wh,
                )
                obj_metrics["obj_source_id"] = 1.0
            else:
                labels = self.object_metrics_engine.get_detector_labels(meta)
                gt_dets = self.object_metrics_engine.run_open_vocab_detection(gt_path, labels)
                pred_dets = self.object_metrics_engine.run_open_vocab_detection(str(pred_img_path), labels)
                obj_metrics = self.object_metrics_engine.compute_from_detections(gt_dets, pred_dets)

            if obj_source == "auto":
                ann_f1 = obj_metrics.get("obj_f1", float("nan"))
                if not np.isfinite(ann_f1):
                    labels = self.object_metrics_engine.get_detector_labels(meta)
                    gt_dets = self.object_metrics_engine.run_open_vocab_detection(gt_path, labels)
                    pred_dets = self.object_metrics_engine.run_open_vocab_detection(str(pred_img_path), labels)
                    det_metrics = self.object_metrics_engine.compute_from_detections(gt_dets, pred_dets)
                    if np.isfinite(det_metrics.get("obj_f1", float("nan"))):
                        obj_metrics = det_metrics

            result.update(obj_metrics)

        if do_vis and vis_save_path:
            try:
                p1, _, _ = self.resolve_frame(meta, item["context"][0])
                p2, _, _ = self.resolve_frame(meta, item["context"][-1])
                if not p1:
                    p1 = c_paths[0]
                if not p2:
                    p2 = c_paths[-1]
                vis_ctx_paths = [p1, p2]

                gt_target_pos = gt_pose[:3, 3]
                pred_target_pos = fused_pose[:3, 3]
                gt_target_R = gt_pose[:3, :3]
                pred_target_R = fused_pose[:3, :3]

                instr_text = item.get("instruction", "N/A")
                metrics_text = (
                    f"TErr: {result.get('pose_t_err_m', float('nan')):.3f} m\n"
                    f"RErr: {result.get('pose_r_err_deg', float('nan')):.2f} deg\n"
                    f"ObjF1: {result.get('obj_f1', float('nan')):.3f}"
                )

                create_combined_visualization(
                    vis_ctx_paths,
                    gt_path,
                    str(pred_img_path),
                    gt_context_poses[:, :3, 3],
                    fused["aligned_pred_centers_ctx"],
                    gt_target_pos,
                    pred_target_pos,
                    gt_target_R,
                    pred_target_R,
                    instr_text,
                    metrics_text,
                    vis_save_path,
                )
            except Exception:
                pass

        return result

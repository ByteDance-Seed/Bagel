from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..config import CONFIG
from ..optional_deps import load_and_preprocess_images, pose_encoding_to_extri_intri
from ..utils import (
    as_4x4,
    build_pointcloud_from_depth,
    convert_to_hwc3,
    ensure_numpy,
    extract_intrinsics_3x3,
    transform_points,
)


class VGGTBackend:
    def __init__(self, model: Any, device: str, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype

    def run_inference(self, image_paths: List[str], need_points: bool = False) -> Optional[Dict[str, Any]]:
        if self.model is None:
            return None
        if load_and_preprocess_images is None or pose_encoding_to_extri_intri is None:
            return None

        try:
            images = load_and_preprocess_images(image_paths).to(self.device)[None]
        except Exception:
            return None

        try:
            with torch.no_grad():
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    ctx = torch.cuda.amp.autocast(dtype=self.dtype)
                else:
                    ctx = nullcontext()

                with ctx:
                    tokens, _ = self.model.aggregator(images)
                    pose_enc = self.model.camera_head(tokens)[-1]
                    extrinsic, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

                    extrinsic_np = ensure_numpy(extrinsic)
                    intri_np = ensure_numpy(intri)

                    if extrinsic_np is None:
                        return None

                    if extrinsic_np.ndim == 4 and extrinsic_np.shape[0] == 1:
                        extrinsic_np = extrinsic_np[0]

                    pred_c2ws = []
                    for i in range(extrinsic_np.shape[0]):
                        m = as_4x4(extrinsic_np[i])
                        pred_c2ws.append(np.linalg.inv(m))
                    pred_c2ws = np.asarray(pred_c2ws)

                    result = {
                        "c2ws": pred_c2ws,
                        "paths": list(image_paths),
                        "path_to_pose": {p: pred_c2ws[i] for i, p in enumerate(image_paths)},
                        "intrinsics": intri_np,
                        "image_hw": (images.shape[-2], images.shape[-1]),
                        "pointclouds_world": None,
                        "target_points_world": np.zeros((0, 3), dtype=np.float32),
                        "backend_used": "vggt",
                    }

                    if need_points:
                        pointclouds = self._extract_pointclouds(tokens, pred_c2ws, intri_np)
                        result["pointclouds_world"] = pointclouds
                        if pointclouds is not None and len(pointclouds) > 0:
                            result["target_points_world"] = pointclouds[-1]

                    return result
        except Exception:
            return None

    def _extract_pointclouds(
        self,
        tokens: torch.Tensor,
        pred_c2ws: np.ndarray,
        intri_np: Optional[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        n_views = pred_c2ws.shape[0]

        if hasattr(self.model, "point_head"):
            try:
                point_out = self.model.point_head(tokens)
                point_hwc3 = convert_to_hwc3(point_out, n_views)
                if point_hwc3 is not None:
                    pts_world_all = []
                    arr = point_hwc3.detach().float().cpu().numpy()
                    for vid in range(min(arr.shape[0], n_views)):
                        p = arr[vid]
                        p = p[:: CONFIG["POINT_STRIDE"], :: CONFIG["POINT_STRIDE"], :].reshape(-1, 3)
                        valid = np.all(np.isfinite(p), axis=1)
                        p = p[valid]
                        if p.shape[0] > 0:
                            z_pos_ratio = float((p[:, 2] > 0).mean())
                        else:
                            z_pos_ratio = 0.0
                        if z_pos_ratio > 0.5:
                            p = transform_points(p, pred_c2ws[vid])
                        pts_world_all.append(p.astype(np.float32))
                    if len(pts_world_all) == n_views:
                        return pts_world_all
            except Exception:
                pass

        if hasattr(self.model, "depth_head"):
            try:
                depth_out = self.model.depth_head(tokens)
                if isinstance(depth_out, (list, tuple)):
                    depth_out = depth_out[-1]
                if torch.is_tensor(depth_out):
                    d = depth_out
                    if d.dim() == 5 and d.shape[0] == 1:
                        d = d[0]
                    if d.dim() == 4 and d.shape[1] == 1:
                        d = d[:, 0]
                    if d.dim() == 3 and d.shape[0] >= n_views:
                        d_np = d.detach().float().cpu().numpy()
                        out_pc = []
                        for vid in range(n_views):
                            K = extract_intrinsics_3x3(intri_np, vid)
                            if K is None:
                                out_pc.append(np.zeros((0, 3), dtype=np.float32))
                                continue
                            pts_cam = build_pointcloud_from_depth(d_np[vid], K, stride=CONFIG["POINT_STRIDE"])
                            pts_world = transform_points(pts_cam, pred_c2ws[vid])
                            out_pc.append(pts_world.astype(np.float32))
                        return out_pc
            except Exception:
                pass

        return None

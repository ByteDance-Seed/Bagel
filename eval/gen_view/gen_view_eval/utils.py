# Geometry, image, and numerical helpers.

import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import PIXEL_GROUP
from .optional_deps import skimage_ssim

def safe_float(v: Any) -> float:
    try:
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def wrap_to_pi(rad: float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_diff = R1 @ R2.T
    val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def robust_mean(arr: Sequence[float]) -> float:
    vals = [x for x in arr if np.isfinite(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def robust_median(arr: Sequence[float]) -> float:
    vals = [x for x in arr if np.isfinite(x)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def as_4x4(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = m
        return out
    raise ValueError(f"Unexpected matrix shape: {m.shape}")


def parse_frame_idx(fname: str) -> int:
    try:
        name_part = os.path.splitext(fname)[0]
        if "-" in name_part:
            name_part = name_part.split("-")[-1]
        return int(name_part)
    except Exception:
        return -1


def ensure_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x)
        except Exception:
            return None
    return None


def convert_to_hwc3(t: torch.Tensor, n_views: int) -> Optional[torch.Tensor]:
    """
    Try to convert unknown tensor layout to [V, H, W, 3].
    """
    if t is None:
        return None

    if isinstance(t, (list, tuple)):
        for cand in reversed(t):
            out = convert_to_hwc3(cand, n_views)
            if out is not None:
                return out
        return None

    if not torch.is_tensor(t):
        return None

    x = t
    if x.dim() == 6 and x.shape[0] == 1:
        x = x[0]

    if x.dim() == 5 and x.shape[0] == 1:
        x = x[0]

    if x.dim() == 5 and x.shape[-1] == 3:
        # [V, H, W, 3] with extra dim likely batch-less already
        return x

    if x.dim() == 5 and x.shape[1] == 3:
        # [V, 3, H, W] -> [V, H, W, 3]
        return x.permute(0, 2, 3, 1)

    if x.dim() == 4:
        if x.shape[-1] == 3 and x.shape[0] == n_views:
            return x
        if x.shape[1] == 3 and x.shape[0] == n_views:
            return x.permute(0, 2, 3, 1)

    return None


def extract_intrinsics_3x3(intri: Any, view_idx: int) -> Optional[np.ndarray]:
    arr = ensure_numpy(intri)
    if arr is None:
        return None

    # Common shapes: [B,V,3,3], [V,3,3], [B,V,4,4], [V,4,4]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        return None

    if view_idx >= arr.shape[0]:
        return None

    K = arr[view_idx]
    if K.shape == (4, 4):
        return K[:3, :3]
    if K.shape == (3, 3):
        return K
    return None


def build_pointcloud_from_depth(depth: np.ndarray, K: np.ndarray, stride: int = 4) -> np.ndarray:
    h, w = depth.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    z = depth[grid_y, grid_x]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    valid = np.isfinite(z) & (z > 1e-4)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (grid_x[valid] - cx) * z[valid] / max(fx, 1e-8)
    y = (grid_y[valid] - cy) * z[valid] / max(fy, 1e-8)
    pts = np.stack([x, y, z[valid]], axis=1).astype(np.float32)
    return pts


def transform_points(pts: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (pts @ R.T) + t[None, :]


def sample_points(pts: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    if pts.shape[0] <= max_n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    return pts[idx]


def chamfer_and_fscore(
    pc1: np.ndarray,
    pc2: np.ndarray,
    f_thresh: float = 0.05,
    max_samples: int = 4096,
    seed: int = 0,
) -> Dict[str, float]:
    if pc1.size == 0 or pc2.size == 0:
        return {"pc_chamfer": float("nan"), "pc_fscore": float("nan")}

    p1 = sample_points(pc1, max_samples, seed)
    p2 = sample_points(pc2, max_samples, seed + 1)

    t1 = torch.from_numpy(p1).float()
    t2 = torch.from_numpy(p2).float()

    with torch.no_grad():
        d = torch.cdist(t1[None], t2[None], p=2)[0]
        d12 = d.min(dim=1).values
        d21 = d.min(dim=0).values

    chamfer = float(d12.mean().item() + d21.mean().item())
    precision = float((d12 < f_thresh).float().mean().item())
    recall = float((d21 < f_thresh).float().mean().item())
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    return {"pc_chamfer": chamfer, "pc_fscore": float(fscore)}


def project_points(points_world: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (uv, z_cam).
    uv: [N,2], z_cam: [N]
    """
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam = (points_world @ R.T) + t[None, :]
    z = cam[:, 2]

    x = cam[:, 0] / np.clip(z, 1e-8, None)
    y = cam[:, 1] / np.clip(z, 1e-8, None)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * x + cx
    v = fy * y + cy
    uv = np.stack([u, v], axis=1)
    return uv, z


def image_metrics(pred_path: str, gt_path: str) -> Dict[str, float]:
    try:
        pred = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return {k: float("nan") for k in PIXEL_GROUP}

    if pred.shape != gt.shape:
        try:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
            pred = np.asarray(pred_img, dtype=np.float32) / 255.0
        except Exception:
            return {k: float("nan") for k in PIXEL_GROUP}

    diff = pred - gt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mse = float(np.mean(diff * diff))
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))

    if skimage_ssim is not None:
        try:
            ssim = float(skimage_ssim(gt, pred, channel_axis=2, data_range=1.0))
        except Exception:
            ssim = float("nan")
    else:
        ssim = float("nan")

    return {
        "pix_mae": mae,
        "pix_rmse": rmse,
        "pix_psnr": psnr,
        "pix_ssim": ssim,
    }


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def run_cmd(cmd: List[str], timeout_s: int = 300) -> bool:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False)
        return proc.returncode == 0
    except Exception:
        return False

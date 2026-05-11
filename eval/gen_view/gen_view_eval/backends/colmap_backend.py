from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import CONFIG
from ..utils import qvec2rotmat, run_cmd


class ColmapBackend:
    def __init__(self, colmap_bin: Optional[str] = None):
        self.colmap_bin = colmap_bin or shutil.which("colmap")

    def _tmp_root(self) -> str:
        configured = str(CONFIG.get("COLMAP_TMP_ROOT", "")).strip()
        if configured:
            Path(configured).mkdir(parents=True, exist_ok=True)
            return configured
        root = str(Path(CONFIG["INPUT_DIR"]) / ".colmap_eval_cache")
        Path(root).mkdir(parents=True, exist_ok=True)
        return root

    def _parse_images_txt(self, images_txt: str) -> Tuple[Dict[str, np.ndarray], Dict[int, str], Dict[str, Set[int]]]:
        poses_by_name: Dict[str, np.ndarray] = {}
        id_to_name: Dict[int, str] = {}
        name_to_point_ids: Dict[str, Set[int]] = {}

        if not os.path.exists(images_txt):
            return poses_by_name, id_to_name, name_to_point_ids

        with open(images_txt, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]

        valid_lines = [ln for ln in lines if ln and not ln.startswith("#")]
        i = 0
        while i < len(valid_lines):
            head = valid_lines[i]
            i += 1
            if i >= len(valid_lines):
                break
            pts_line = valid_lines[i]
            i += 1

            toks = head.split()
            if len(toks) < 10:
                continue
            try:
                image_id = int(toks[0])
                qvec = np.array([float(x) for x in toks[1:5]], dtype=np.float64)
                tvec = np.array([float(x) for x in toks[5:8]], dtype=np.float64)
                name = toks[9]
            except Exception:
                continue

            R = qvec2rotmat(qvec)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = tvec
            c2w = np.linalg.inv(w2c)

            poses_by_name[name] = c2w
            id_to_name[image_id] = name

            point_ids: Set[int] = set()
            ptoks = pts_line.split()
            for j in range(2, len(ptoks), 3):
                try:
                    pid = int(float(ptoks[j]))
                except Exception:
                    continue
                if pid > 0:
                    point_ids.add(pid)
            name_to_point_ids[name] = point_ids

        return poses_by_name, id_to_name, name_to_point_ids

    def _parse_points3d_txt(self, points_txt: str) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        if not os.path.exists(points_txt):
            return out
        with open(points_txt, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                toks = ln.split()
                if len(toks) < 4:
                    continue
                try:
                    pid = int(toks[0])
                    xyz = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float32)
                    out[pid] = xyz
                except Exception:
                    continue
        return out

    def run_inference(
        self,
        image_paths: List[str],
        target_abs_path: str,
        cam2img: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.colmap_bin is None:
            return None
        if len(image_paths) < 2:
            return None

        try:
            key = "||".join(os.path.abspath(p) for p in image_paths)
            key += f"||{CONFIG.get('COLMAP_MATCHER', 'sequential')}||{CONFIG.get('COLMAP_MAX_NUM_FEATURES', 8192)}"
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

            root = self._tmp_root()
            workdir = os.path.join(root, digest)
            image_dir = os.path.join(workdir, "images")
            sparse_dir = os.path.join(workdir, "sparse")
            txt_dir = os.path.join(workdir, "sparse_txt")
            db_path = os.path.join(workdir, "database.db")

            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(sparse_dir, exist_ok=True)

            local_names = []
            for i, src in enumerate(image_paths):
                ext = Path(src).suffix if Path(src).suffix else ".jpg"
                name = f"{i:03d}{ext.lower()}"
                dst = os.path.join(image_dir, name)
                local_names.append(name)
                if not os.path.exists(dst):
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except Exception:
                        shutil.copy2(src, dst)

            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except Exception:
                    pass
            if os.path.isdir(sparse_dir):
                for child in Path(sparse_dir).glob("*"):
                    if child.is_dir():
                        shutil.rmtree(str(child), ignore_errors=True)

            use_gpu = "1" if bool(CONFIG.get("COLMAP_USE_GPU", True)) else "0"
            max_feat = str(int(CONFIG.get("COLMAP_MAX_NUM_FEATURES", 8192)))

            feat_cmd = [
                self.colmap_bin,
                "feature_extractor",
                "--database_path",
                db_path,
                "--image_path",
                image_dir,
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.use_gpu",
                use_gpu,
                "--SiftExtraction.max_num_features",
                max_feat,
            ]
            if cam2img is not None:
                K = np.asarray(cam2img)
                if K.shape == (4, 4):
                    K = K[:3, :3]
                if K.shape == (3, 3):
                    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
                    feat_cmd += ["--ImageReader.camera_model", "PINHOLE"]
                    feat_cmd += ["--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}"]

            if not run_cmd(feat_cmd, timeout_s=600):
                return None

            matcher = str(CONFIG.get("COLMAP_MATCHER", "sequential")).lower()
            if matcher == "exhaustive":
                match_cmd = [
                    self.colmap_bin,
                    "exhaustive_matcher",
                    "--database_path",
                    db_path,
                    "--SiftMatching.use_gpu",
                    use_gpu,
                ]
            else:
                overlap = str(int(CONFIG.get("COLMAP_SEQUENTIAL_OVERLAP", 25)))
                match_cmd = [
                    self.colmap_bin,
                    "sequential_matcher",
                    "--database_path",
                    db_path,
                    "--SiftMatching.use_gpu",
                    use_gpu,
                    "--SequentialMatching.overlap",
                    overlap,
                    "--SequentialMatching.loop_detection",
                    "0",
                ]
            if not run_cmd(match_cmd, timeout_s=600):
                return None

            map_cmd = [
                self.colmap_bin,
                "mapper",
                "--database_path",
                db_path,
                "--image_path",
                image_dir,
                "--output_path",
                sparse_dir,
                "--Mapper.multiple_models",
                "0",
            ]
            if not run_cmd(map_cmd, timeout_s=900):
                return None

            model_dirs = [p for p in Path(sparse_dir).glob("*") if p.is_dir()]
            if not model_dirs:
                return None
            model_dir = sorted(model_dirs, key=lambda p: p.name)[0]

            os.makedirs(txt_dir, exist_ok=True)
            convert_cmd = [
                self.colmap_bin,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                txt_dir,
                "--output_type",
                "TXT",
            ]
            if not run_cmd(convert_cmd, timeout_s=300):
                return None

            poses_by_name, _id_to_name, name_to_point_ids = self._parse_images_txt(os.path.join(txt_dir, "images.txt"))
            if not poses_by_name:
                return None
            points_map = self._parse_points3d_txt(os.path.join(txt_dir, "points3D.txt"))

            c2ws: List[np.ndarray] = []
            used_paths: List[str] = []
            for i, src in enumerate(image_paths):
                name = local_names[i]
                pose = poses_by_name.get(name)
                if pose is None:
                    continue
                c2ws.append(pose)
                used_paths.append(src)

            if not c2ws:
                return None

            target_idx = len(image_paths) - 1
            abs_target = os.path.abspath(target_abs_path)
            for i, p in enumerate(image_paths):
                if os.path.abspath(p) == abs_target:
                    target_idx = i
                    break
            target_name = local_names[target_idx]
            target_point_ids = name_to_point_ids.get(target_name, set())
            target_pts = []
            for pid in target_point_ids:
                xyz = points_map.get(pid)
                if xyz is not None and np.all(np.isfinite(xyz)):
                    target_pts.append(xyz)
            if target_pts:
                target_points = np.stack(target_pts, axis=0).astype(np.float32)
            else:
                target_points = np.zeros((0, 3), dtype=np.float32)

            return {
                "c2ws": np.asarray(c2ws),
                "paths": used_paths,
                "path_to_pose": {p: c2ws[i] for i, p in enumerate(used_paths)},
                "target_points_world": target_points,
                "backend_used": "colmap",
            }
        except Exception:
            return None

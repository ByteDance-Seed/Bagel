from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

from ..config import CONFIG
from ..optional_deps import AutoProcessor, Owlv2ForObjectDetection
from ..utils import bbox_iou_xyxy, project_points, robust_mean, wrap_to_pi


class ObjectMetricsEngine:
    def __init__(self, device: str):
        self.device = device
        self.detector_processor = None
        self.detector_model = None
        self.detector_device = "cpu"

    def empty_metrics(self) -> Dict[str, float]:
        return {
            "obj_precision": float("nan"),
            "obj_recall": float("nan"),
            "obj_f1": float("nan"),
            "obj_iou": float("nan"),
            "obj_rel_acc": float("nan"),
            "obj_orient_err_deg": float("nan"),
            "obj_gt_count": float("nan"),
            "obj_pred_count": float("nan"),
            "obj_intersection_count": float("nan"),
            "obj_source_id": 0.0,
        }

    def _estimate_visible_instances(
        self,
        instances: Dict[int, Dict[str, Any]],
        pose_c2w: np.ndarray,
        K: np.ndarray,
        image_wh: Tuple[int, int],
    ) -> Tuple[Set[int], Dict[int, Dict[str, Any]]]:
        w, h = image_wh
        if K.shape == (4, 4):
            K3 = K[:3, :3]
        else:
            K3 = K

        pred_ids: Set[int] = set()
        per_obj: Dict[int, Dict[str, Any]] = {}

        if not instances:
            return pred_ids, per_obj

        obj_ids = list(instances.keys())
        centers = np.stack([instances[i]["center"] for i in obj_ids], axis=0)
        uv, z = project_points(centers, pose_c2w, K3)

        for i, obj_id in enumerate(obj_ids):
            u, v = uv[i]
            zz = z[i]
            inside = bool((zz > 1e-4) and (0 <= u < w) and (0 <= v < h))
            if inside:
                pred_ids.add(obj_id)

            per_obj[obj_id] = {
                "uv": np.array([u, v], dtype=np.float64),
                "z": float(zz),
                "inside": inside,
                "yaw": float(instances[obj_id]["yaw"]),
                "center": instances[obj_id]["center"],
            }

        return pred_ids, per_obj

    @staticmethod
    def _camera_yaw(c2w: np.ndarray) -> float:
        forward = c2w[:3, 2]
        return float(np.arctan2(forward[1], forward[0]))

    def compute_from_annotations(
        self,
        instances: Dict[int, Dict[str, Any]],
        gt_visible_ids: Set[int],
        gt_pose: np.ndarray,
        pred_pose: np.ndarray,
        K: np.ndarray,
        image_wh: Tuple[int, int],
    ) -> Dict[str, float]:
        pred_ids, pred_info = self._estimate_visible_instances(instances, pred_pose, K, image_wh)
        gt_proj_ids, gt_info = self._estimate_visible_instances(instances, gt_pose, K, image_wh)

        if gt_visible_ids:
            gt_ids = set(int(x) for x in gt_visible_ids)
        else:
            gt_ids = gt_proj_ids

        inter = pred_ids & gt_ids
        union = pred_ids | gt_ids

        precision = len(inter) / max(len(pred_ids), 1)
        recall = len(inter) / max(len(gt_ids), 1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = len(inter) / max(len(union), 1)

        shared = sorted(list(inter))
        rel_correct = 0
        rel_total = 0
        eps_xy = float(CONFIG["REL_EPS_PIXEL"])
        eps_z = float(CONFIG["REL_EPS_DEPTH"])

        for i in range(len(shared)):
            for j in range(i + 1, len(shared)):
                a = shared[i]
                b = shared[j]

                g_a = gt_info[a]
                g_b = gt_info[b]
                p_a = pred_info[a]
                p_b = pred_info[b]

                dx_gt = g_a["uv"][0] - g_b["uv"][0]
                dx_pr = p_a["uv"][0] - p_b["uv"][0]
                if abs(dx_gt) > eps_xy:
                    rel_total += 1
                    rel_correct += int(np.sign(dx_gt) == np.sign(dx_pr))

                dy_gt = g_a["uv"][1] - g_b["uv"][1]
                dy_pr = p_a["uv"][1] - p_b["uv"][1]
                if abs(dy_gt) > eps_xy:
                    rel_total += 1
                    rel_correct += int(np.sign(dy_gt) == np.sign(dy_pr))

                dz_gt = g_a["z"] - g_b["z"]
                dz_pr = p_a["z"] - p_b["z"]
                if abs(dz_gt) > eps_z:
                    rel_total += 1
                    rel_correct += int(np.sign(dz_gt) == np.sign(dz_pr))

        rel_acc = rel_correct / max(rel_total, 1)

        gt_cam_yaw = self._camera_yaw(gt_pose)
        pred_cam_yaw = self._camera_yaw(pred_pose)
        orient_errs = []
        for obj_id in shared:
            yaw_obj = float(instances[obj_id]["yaw"])
            rel_gt = wrap_to_pi(yaw_obj - gt_cam_yaw)
            rel_pr = wrap_to_pi(yaw_obj - pred_cam_yaw)
            orient_errs.append(abs(np.degrees(wrap_to_pi(rel_pr - rel_gt))))

        orient_err = robust_mean(orient_errs)

        return {
            "obj_precision": float(precision),
            "obj_recall": float(recall),
            "obj_f1": float(f1),
            "obj_iou": float(iou),
            "obj_rel_acc": float(rel_acc),
            "obj_orient_err_deg": float(orient_err),
            "obj_gt_count": float(len(gt_ids)),
            "obj_pred_count": float(len(pred_ids)),
            "obj_intersection_count": float(len(inter)),
        }

    def get_detector_labels(self, meta: Dict[str, Any]) -> List[str]:
        labels: List[str] = []

        if meta.get("instances"):
            seen = set()
            for inst in meta["instances"].values():
                name = str(inst.get("label_name", "")).strip().lower()
                if not name or name.isdigit() or len(name) > 50:
                    continue
                if name not in seen:
                    labels.append(name)
                    seen.add(name)

        cfg_labels = [str(x).strip().lower() for x in CONFIG.get("DETECTOR_LABELS", []) if str(x).strip()]
        for lb in cfg_labels:
            if lb not in labels:
                labels.append(lb)

        if not labels:
            labels = cfg_labels
        return labels

    def _init_detector(self) -> bool:
        if self.detector_model is not None and self.detector_processor is not None:
            return True

        backend = str(CONFIG.get("DETECTOR_BACKEND", "owlv2")).lower()
        if backend != "owlv2":
            return False
        if AutoProcessor is None or Owlv2ForObjectDetection is None:
            return False

        model_id = str(CONFIG.get("DETECTOR_MODEL_ID", "google/owlv2-large-patch14-ensemble"))
        try:
            self.detector_processor = AutoProcessor.from_pretrained(model_id)
            self.detector_model = Owlv2ForObjectDetection.from_pretrained(model_id)
            self.detector_device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
            self.detector_model.to(self.detector_device)
            self.detector_model.eval()
            return True
        except Exception:
            self.detector_model = None
            self.detector_processor = None
            return False

    def run_open_vocab_detection(self, image_path: str, labels: List[str]) -> Optional[List[Dict[str, Any]]]:
        if not labels:
            return []
        if not self._init_detector():
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            query = [labels]
            inputs = self.detector_processor(text=query, images=img, return_tensors="pt")
            for k, v in list(inputs.items()):
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.detector_device)

            with torch.no_grad():
                outputs = self.detector_model(**inputs)

            target_sizes = torch.tensor([img.size[::-1]], device=self.detector_device)
            processed = self.detector_processor.post_process_object_detection(
                outputs=outputs,
                threshold=float(CONFIG.get("DETECTOR_SCORE_THRESH", 0.15)),
                target_sizes=target_sizes,
            )
            det = processed[0]

            boxes = det["boxes"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            label_ids = det["labels"].detach().cpu().numpy()

            out: List[Dict[str, Any]] = []
            for box, score, lid in zip(boxes, scores, label_ids):
                lid_int = int(lid)
                if lid_int < 0 or lid_int >= len(labels):
                    continue
                x1, y1, x2, y2 = [float(x) for x in box]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                out.append(
                    {
                        "label": labels[lid_int],
                        "score": float(score),
                        "box": np.array([x1, y1, x2, y2], dtype=np.float64),
                        "center": np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64),
                        "area": float(w * h),
                    }
                )
            return out
        except Exception:
            return None

    def compute_from_detections(self, gt_dets: List[Dict[str, Any]], pred_dets: List[Dict[str, Any]]) -> Dict[str, float]:
        if gt_dets is None or pred_dets is None:
            return self.empty_metrics()

        iou_th = float(CONFIG.get("DETECTOR_IOU_MATCH_THRESH", 0.3))
        cand = []
        for pi, p in enumerate(pred_dets):
            for gi, g in enumerate(gt_dets):
                if p["label"] != g["label"]:
                    continue
                iou = bbox_iou_xyxy(p["box"], g["box"])
                if iou >= iou_th:
                    cand.append((iou, pi, gi))
        cand.sort(key=lambda x: x[0], reverse=True)

        used_p = set()
        used_g = set()
        matches: List[Tuple[int, int, float]] = []
        for iou, pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            matches.append((pi, gi, iou))

        tp = len(matches)
        precision = tp / max(len(pred_dets), 1)
        recall = tp / max(len(gt_dets), 1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / max(len(pred_dets) + len(gt_dets) - tp, 1)

        rel_correct = 0
        rel_total = 0
        eps_xy = float(CONFIG["REL_EPS_PIXEL"])
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                p1, g1, _ = matches[i]
                p2, g2, _ = matches[j]

                gt_a = gt_dets[g1]
                gt_b = gt_dets[g2]
                pr_a = pred_dets[p1]
                pr_b = pred_dets[p2]

                dx_gt = gt_a["center"][0] - gt_b["center"][0]
                dx_pr = pr_a["center"][0] - pr_b["center"][0]
                if abs(dx_gt) > eps_xy:
                    rel_total += 1
                    rel_correct += int(np.sign(dx_gt) == np.sign(dx_pr))

                dy_gt = gt_a["center"][1] - gt_b["center"][1]
                dy_pr = pr_a["center"][1] - pr_b["center"][1]
                if abs(dy_gt) > eps_xy:
                    rel_total += 1
                    rel_correct += int(np.sign(dy_gt) == np.sign(dy_pr))

                da_gt = gt_a["area"] - gt_b["area"]
                da_pr = pr_a["area"] - pr_b["area"]
                if abs(da_gt) > 1e-6:
                    rel_total += 1
                    rel_correct += int(np.sign(da_gt) == np.sign(da_pr))

        rel_acc = rel_correct / max(rel_total, 1)

        return {
            "obj_precision": float(precision),
            "obj_recall": float(recall),
            "obj_f1": float(f1),
            "obj_iou": float(iou),
            "obj_rel_acc": float(rel_acc),
            "obj_orient_err_deg": float("nan"),
            "obj_gt_count": float(len(gt_dets)),
            "obj_pred_count": float(len(pred_dets)),
            "obj_intersection_count": float(tp),
            "obj_source_id": 2.0,
        }

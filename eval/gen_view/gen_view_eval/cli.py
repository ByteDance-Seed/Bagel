# Command line entrypoint and reporting.

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import CONFIG, OBJECT_GROUP, PIXEL_GROUP, POINT_GROUP, POSE_GROUP
from .evaluator import ScanNetEvaluator
from .optional_deps import VGGT
from .utils import safe_float

def summarize_metrics(df: pd.DataFrame, keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        if k not in df.columns:
            out[k] = float("nan")
            continue
        vals = pd.to_numeric(df[k], errors="coerce")
        out[k] = safe_float(vals.mean(skipna=True))
    return out


def print_group(name: str, summary: Dict[str, float]):
    print(f"\n[{name}]")
    for k, v in summary.items():
        if np.isnan(v):
            print(f"  {k}: nan")
        else:
            print(f"  {k}: {v:.6f}")


# ==========================================
# Main
# ==========================================
def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate novel-view generation with pose/pixel/point/object metrics.")
    parser.add_argument("--input_dir", type=str, default=CONFIG["INPUT_DIR"])
    parser.add_argument("--vis_count", type=int, default=CONFIG["VIS_COUNT"])
    parser.add_argument("--disable_point", action="store_true")
    parser.add_argument("--disable_pixel", action="store_true")
    parser.add_argument("--disable_object", action="store_true")
    parser.add_argument("--pose_backend", type=str, default=CONFIG["POSE_BACKEND"], choices=["auto", "colmap", "vggt"])
    parser.add_argument("--pose_prior_blend", type=float, default=CONFIG["POSE_PRIOR_BLEND"])
    parser.add_argument("--object_source", type=str, default=CONFIG["OBJECT_SOURCE"], choices=["auto", "annotation", "detector"])
    parser.add_argument("--detector_model_id", type=str, default=CONFIG["DETECTOR_MODEL_ID"])
    parser.add_argument("--colmap_matcher", type=str, default=CONFIG["COLMAP_MATCHER"], choices=["sequential", "exhaustive"])
    parser.add_argument("--colmap_use_gpu", type=int, default=1 if CONFIG["COLMAP_USE_GPU"] else 0, choices=[0, 1])
    parser.add_argument("--colmap_tmp_root", type=str, default=CONFIG["COLMAP_TMP_ROOT"])
    parser.add_argument(
        "--detector_labels",
        type=str,
        default="",
        help="Comma separated object vocabulary for open-vocab detector. Empty uses built-in list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    CONFIG["INPUT_DIR"] = args.input_dir
    CONFIG["VIS_COUNT"] = args.vis_count
    CONFIG["ENABLE_POINT_METRICS"] = not args.disable_point
    CONFIG["ENABLE_PIXEL_METRICS"] = not args.disable_pixel
    CONFIG["ENABLE_OBJECT_METRICS"] = not args.disable_object
    CONFIG["POSE_BACKEND"] = args.pose_backend
    CONFIG["POSE_PRIOR_BLEND"] = args.pose_prior_blend
    CONFIG["OBJECT_SOURCE"] = args.object_source
    CONFIG["DETECTOR_MODEL_ID"] = args.detector_model_id
    CONFIG["COLMAP_MATCHER"] = args.colmap_matcher
    CONFIG["COLMAP_USE_GPU"] = bool(args.colmap_use_gpu)
    CONFIG["COLMAP_TMP_ROOT"] = args.colmap_tmp_root
    if args.detector_labels.strip():
        CONFIG["DETECTOR_LABELS"] = [x.strip().lower() for x in args.detector_labels.split(",") if x.strip()]

    random.seed(CONFIG["RANDOM_SEED"])
    np.random.seed(CONFIG["RANDOM_SEED"])

    json_path = Path(CONFIG["INPUT_DIR"]) / "predictions.json"
    if not json_path.exists():
        print(f"predictions.json not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data_list = json.load(f)

    vis_save_dir = Path(CONFIG["INPUT_DIR"]) / CONFIG["VIS_DIR_NAME"]
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {vis_save_dir}")

    if len(data_list) == 0:
        print("No prediction items found.")
        return

    vis_indices = set(np.random.choice(len(data_list), min(CONFIG["VIS_COUNT"], len(data_list)), replace=False))
    print(f"Selected {len(vis_indices)} samples for visualization.")

    vggt = None
    need_vggt = CONFIG["POSE_BACKEND"] in {"auto", "vggt"} or CONFIG["ENABLE_POINT_METRICS"]
    if need_vggt and VGGT is not None:
        print(f"Loading Model: {CONFIG['MODEL_PATH']}")
        try:
            vggt = VGGT.from_pretrained(CONFIG["MODEL_PATH"]).to(CONFIG["DEVICE"])
            vggt.eval()
        except Exception as e:
            print(f"Model load warning: {e}")
            vggt = None

    evaluator = ScanNetEvaluator(vggt, device=CONFIG["DEVICE"])
    if CONFIG["POSE_BACKEND"] == "colmap" and evaluator.colmap_bin is None:
        print("COLMAP backend requested but `colmap` binary is not found in PATH.")
        return
    if CONFIG["POSE_BACKEND"] == "vggt" and vggt is None:
        print("VGGT backend requested but model/runtime is unavailable.")
        return
    if CONFIG["POSE_BACKEND"] == "auto" and evaluator.colmap_bin is None and vggt is None:
        print("Neither COLMAP nor VGGT backend is available. Abort evaluation.")
        return
    print(
        "Backend status:",
        f"pose_policy={CONFIG['POSE_BACKEND']},",
        f"colmap_available={evaluator.colmap_bin is not None},",
        f"vggt_available={vggt is not None}",
    )
    results = []

    print("Starting Evaluation...")
    folder_name = Path(CONFIG["INPUT_DIR"]).name
    print(f"Task folder: {folder_name}")

    for i, item in tqdm(enumerate(data_list), total=len(data_list)):
        do_vis = i in vis_indices
        vis_path = str(vis_save_dir / f"sample_{i:06d}_orientation.jpg") if do_vis else None

        try:
            res = evaluator.evaluate_single_item(item, i, do_vis=do_vis, vis_save_path=vis_path)
            if res:
                results.append(res)
        except KeyboardInterrupt:
            break
        except Exception:
            continue

    if not results:
        print("No valid sample was evaluated.")
        return

    df = pd.DataFrame(results)
    csv_path = Path(CONFIG["INPUT_DIR"]) / "metrics_final.csv"
    df.to_csv(csv_path, index=False)

    pose_summary = summarize_metrics(df, POSE_GROUP)
    pixel_summary = summarize_metrics(df, PIXEL_GROUP)
    point_summary = summarize_metrics(df, POINT_GROUP)
    object_summary = summarize_metrics(df, OBJECT_GROUP)

    print("\n" + "=" * 60)
    print(f"Summary (n={len(df)})")
    print_group("Pose", pose_summary)
    if CONFIG["ENABLE_PIXEL_METRICS"]:
        print_group("Pixel", pixel_summary)
    if CONFIG["ENABLE_POINT_METRICS"]:
        print_group("PointCloud", point_summary)
    if CONFIG["ENABLE_OBJECT_METRICS"]:
        print_group("Object-Relation", object_summary)
    print("=" * 60)

    summary_payload = {
        "num_samples": int(len(df)),
        "pose": pose_summary,
        "pixel": pixel_summary,
        "pointcloud": point_summary,
        "object_relation": object_summary,
        "config": {
            "pose_prior_blend": CONFIG["POSE_PRIOR_BLEND"],
            "pose_backend": CONFIG["POSE_BACKEND"],
            "enable_pixel": CONFIG["ENABLE_PIXEL_METRICS"],
            "enable_point": CONFIG["ENABLE_POINT_METRICS"],
            "enable_object": CONFIG["ENABLE_OBJECT_METRICS"],
            "object_source": CONFIG["OBJECT_SOURCE"],
            "detector_backend": CONFIG["DETECTOR_BACKEND"],
            "detector_model_id": CONFIG["DETECTOR_MODEL_ID"],
            "colmap_matcher": CONFIG["COLMAP_MATCHER"],
            "colmap_use_gpu": CONFIG["COLMAP_USE_GPU"],
        },
    }
    summary_path = Path(CONFIG["INPUT_DIR"]) / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Per-sample metrics saved to: {csv_path}")
    print(f"Grouped summary saved to: {summary_path}")


if __name__ == "__main__":
    plt.switch_backend("agg")
    main()

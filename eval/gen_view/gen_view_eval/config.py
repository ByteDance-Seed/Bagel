# Central evaluation configuration and metric groups.

CONFIG = {
    "INPUT_DIR": "/mnt/inspurfs/efm_t/longyilin/novelview_ego_centric/scannet/outputs_refine_0215/bagel_7b",
    "INFO_DIR": "/mnt/inspurfs/mozi_t/linjingli/mmscan/embodiedscan_info",
    "IMAGE_ROOT": "/mnt/inspurfs/mozi_t/linjingli/transfer/ScanNet_v2",
    "MODEL_PATH": "/mnt/inspurfs/mozi_t/huwenbo/weights/VGGT-1B",
    "NUM_C": 8,
    "DEVICE": "cuda",
    "VIS_COUNT": 20,
    "RANDOM_SEED": 42,
    "VIS_DIR_NAME": "vis_orientation_results",
    # Metric switches
    "ENABLE_PIXEL_METRICS": True,
    "ENABLE_POINT_METRICS": True,
    "ENABLE_OBJECT_METRICS": True,
    # Pose backend: auto | colmap | vggt
    "POSE_BACKEND": "auto",
    # Object metrics source: auto | annotation | detector
    "OBJECT_SOURCE": "auto",
    # Pose prior fusion
    "POSE_PRIOR_BLEND": 0.7,
    "POSE_PRIOR_MIN_CTX": 3,
    # Point cloud metrics
    "POINT_MAX_SAMPLES": 4096,
    "POINT_STRIDE": 4,
    "POINT_F_THRESH": 0.05,
    # Object relation thresholds
    "REL_EPS_PIXEL": 12.0,
    "REL_EPS_DEPTH": 0.15,
    # Detector settings (used when scene lacks 3D object annotations or OBJECT_SOURCE=detector)
    "DETECTOR_BACKEND": "owlv2",
    "DETECTOR_MODEL_ID": "google/owlv2-large-patch14-ensemble",
    "DETECTOR_SCORE_THRESH": 0.15,
    "DETECTOR_IOU_MATCH_THRESH": 0.3,
    "DETECTOR_LABELS": [
        "person",
        "chair",
        "table",
        "sofa",
        "bed",
        "cabinet",
        "desk",
        "shelf",
        "window",
        "door",
        "monitor",
        "tv",
        "laptop",
        "book",
        "plant",
        "lamp",
        "toilet",
        "sink",
        "bathtub",
        "refrigerator",
        "microwave",
        "oven",
        "stove",
        "pillow",
        "curtain",
        "picture",
        "mirror",
        "box",
        "bag",
        "bottle",
        "cup",
        "keyboard",
        "mouse",
        "phone",
    ],
    # COLMAP settings (for high-accuracy pose/geometry when available)
    "COLMAP_MATCHER": "sequential",
    "COLMAP_USE_GPU": True,
    "COLMAP_TMP_ROOT": "",
    "COLMAP_SEQUENTIAL_OVERLAP": 25,
    "COLMAP_MAX_NUM_FEATURES": 8192,
    "COLMAP_MIN_MATCHED_CONTEXT": 6,
}


POSE_GROUP = [
    "pose_t_err_m",
    "pose_r_err_deg",
    "rpe_trans_err_m",
    "rpe_trans_dir_cos",
    "rpe_rot_deg",
    "ctx_align_rmse_m",
    "ctx_scale",
]
PIXEL_GROUP = ["pix_mae", "pix_rmse", "pix_psnr", "pix_ssim"]
POINT_GROUP = ["pc_chamfer", "pc_fscore"]
OBJECT_GROUP = [
    "obj_precision",
    "obj_recall",
    "obj_f1",
    "obj_iou",
    "obj_rel_acc",
    "obj_orient_err_deg",
]

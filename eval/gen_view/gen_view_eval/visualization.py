# Visualization utilities for qualitative inspection.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

def draw_camera_axes(ax, center, rotation, scale=0.3):
    axes = np.eye(3) * scale
    world_axes = np.dot(rotation, axes.T).T
    colors = ["r", "g", "b"]

    for i, c in enumerate(colors):
        ax.plot(
            [center[0], center[0] + world_axes[i, 0]],
            [center[1], center[1] + world_axes[i, 1]],
            [center[2], center[2] + world_axes[i, 2]],
            color=c,
            linewidth=2,
        )


def create_combined_visualization(
    vis_context_paths,
    gt_path,
    pred_path,
    gt_centers_ctx,
    aligned_pred_centers_ctx,
    gt_target_pos,
    aligned_target_pos,
    gt_target_R,
    aligned_target_R,
    instruction,
    metrics_text,
    save_path,
):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 1.2, 3], hspace=0.3)

    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    full_text = f"Instruction: {instruction}\n\nMetrics: {metrics_text}"
    ax_text.text(0.01, 0.5, full_text, fontsize=13, va="center", wrap=True)

    gs_imgs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.1)

    ax_c1 = fig.add_subplot(gs_imgs[0, 0])
    try:
        ax_c1.imshow(Image.open(vis_context_paths[0]).convert("RGB"))
        ax_c1.set_title("Context (Start)", fontsize=11)
    except Exception:
        ax_c1.text(0.5, 0.5, "Err")
    ax_c1.axis("off")

    ax_c2 = fig.add_subplot(gs_imgs[0, 1])
    try:
        ax_c2.imshow(Image.open(vis_context_paths[1]).convert("RGB"))
        ax_c2.set_title("Context (End)", fontsize=11)
    except Exception:
        ax_c2.text(0.5, 0.5, "Err")
    ax_c2.axis("off")

    ax_gt = fig.add_subplot(gs_imgs[0, 2])
    try:
        ax_gt.imshow(Image.open(gt_path).convert("RGB"))
        ax_gt.set_title("GT Target", fontsize=12, color="blue", fontweight="bold")
    except Exception:
        pass
    ax_gt.axis("off")

    ax_pred = fig.add_subplot(gs_imgs[0, 3])
    try:
        ax_pred.imshow(Image.open(pred_path).convert("RGB"))
        ax_pred.set_title("Pred Target", fontsize=12, color="red", fontweight="bold")
    except Exception:
        pass
    ax_pred.axis("off")

    ax_3d = fig.add_subplot(gs[2], projection="3d")

    ax_3d.plot(
        gt_centers_ctx[:, 0],
        gt_centers_ctx[:, 1],
        gt_centers_ctx[:, 2],
        c="blue",
        label="GT Traj",
        linewidth=1,
        marker=".",
        markersize=2,
        alpha=0.3,
    )
    ax_3d.plot(
        aligned_pred_centers_ctx[:, 0],
        aligned_pred_centers_ctx[:, 1],
        aligned_pred_centers_ctx[:, 2],
        c="red",
        label="Pred Traj (Aligned)",
        linewidth=1,
        linestyle="--",
        marker=".",
        markersize=2,
        alpha=0.3,
    )

    ax_3d.scatter(*gt_target_pos, c="blue", marker="o", s=50, label="GT Pos")
    ax_3d.scatter(*aligned_target_pos, c="red", marker="o", s=50, label="Pred Pos")

    traj_span = np.ptp(gt_centers_ctx, axis=0).max()
    axis_scale = max(0.2, traj_span * 0.1)

    draw_camera_axes(ax_3d, gt_target_pos, gt_target_R, scale=axis_scale)
    draw_camera_axes(ax_3d, aligned_target_pos, aligned_target_R, scale=axis_scale)

    ax_3d.plot(
        [gt_target_pos[0], aligned_target_pos[0]],
        [gt_target_pos[1], aligned_target_pos[1]],
        [gt_target_pos[2], aligned_target_pos[2]],
        c="gray",
        linestyle=":",
        alpha=0.5,
    )

    ax_3d.set_title("3D Alignment & Orientation", fontsize=12)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.legend()

    all_p = np.vstack([gt_centers_ctx, gt_target_pos, aligned_target_pos])
    mid = np.mean(all_p, axis=0)
    max_range = np.max(np.ptp(all_p, axis=0)) / 2.0
    ax_3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax_3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax_3d.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)

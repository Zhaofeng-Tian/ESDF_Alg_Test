import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d


def main():
    # =========================================================
    # USER CONFIG
    # =========================================================
    # image_path = "basement.png"     # <-- your map image
    image_path = "map2.png"
    resolution = 0.01               # meters / pixel
    max_dist = 5.0                  # ESDF truncation (meters)

    obstacle_is_dark = True         # black = obstacle
    thresh = 200

    # ESDF visualization range
    vmin = -1.0                     # inside obstacle
    vmax = 4.0                      # very safe

    # =========================================================
    # 1) LOAD OCCUPANCY
    # =========================================================
    obstacle_mask = load_occupancy_from_png(
        image_path,
        obstacle_is_dark=obstacle_is_dark,
        thresh=thresh,
    )

    print("[INFO] Map loaded")
    print("  shape:", obstacle_mask.shape)
    print("  obstacle ratio:", obstacle_mask.mean())

    # =========================================================
    # 2) COMPUTE SIGNED ESDF
    # =========================================================
    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=max_dist,
    )

    print("[INFO] ESDF computed")
    print("  min sdf:", sdf.min())
    print("  max sdf:", sdf.max())

    # =========================================================
    # 3) VISUALIZATION
    # =========================================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Obstacle mask
    axs[0].imshow(obstacle_mask, cmap="gray")
    axs[0].set_title("Obstacle mask")
    axs[0].axis("off")

    # --- ESDF heatmap
    sdf_vis = sdf.copy()
    sdf_vis[obstacle_mask] = vmin   # force obstacles to pure red

    im = axs[1].imshow(
        sdf_vis,
        cmap="RdYlGn",               # red -> yellow -> green
        vmin=vmin,
        vmax=vmax,
    )

    axs[1].set_title("Signed ESDF (m)\nred=inside obstacle, green=safer")
    axs[1].axis("off")

    cbar = plt.colorbar(im, ax=axs[1], fraction=0.046)
    cbar.set_label("Signed distance (m)")
    cbar.set_ticks([-1, 0, 1, 2, 3, 4])

    plt.tight_layout()
    plt.show()

    # =========================================================
    # 4) SANITY CHECKS (IMPORTANT)
    # =========================================================
    if obstacle_mask.any():
        print("[CHECK] inside obstacle max (should <= 0):",
              sdf[obstacle_mask].max())
    if (~obstacle_mask).any():
        print("[CHECK] free space min (should >= 0):",
              sdf[~obstacle_mask].min())

"""
387 x 903



"""
if __name__ == "__main__":
    main()

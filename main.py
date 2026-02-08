# main.py
from __future__ import annotations
import numpy as np

from preprocess import load_occupancy_from_png, inflate_obstacles, make_origin_xy
from esdf import signed_esdf_2d
from optimizer import chompish_optimize_xy
from car_model import headings_from_path
from simulate import plot_result, animate_history

def main():
    # -------- User settings --------
    png_path = "map.png"        # <-- set your PNG path
    resolution = 0.05           # meters per pixel
    obstacle_is_dark = True
    thresh = 200

    # If you want to inflate obstacles using robot radius:
    # inflation_radius_px ≈ robot_radius_m / resolution
    robot_radius_m = 0.25
    inflation_radius_px = 0  # set e.g. int(robot_radius_m / resolution)

    max_dist = 4.0             # ESDF truncation distance
    origin_mode = "center"     # "center" or "zero"

    # start/goal in world meters
    p0 = np.array([-1.6, -1.2], dtype=float)
    pN = np.array([ 1.6,  1.2], dtype=float)

    # optimizer params
    N = 100
    iters = 200
    alpha = 0.03
    w_obs = 20.0
    safe_dist_m = 0.45
    grad_clip = 5.0

    do_animation = True
    # --------------------------------

    obstacle_mask = load_occupancy_from_png(
        png_path, obstacle_is_dark=obstacle_is_dark, thresh=thresh
    )
    obstacle_mask = inflate_obstacles(obstacle_mask, inflation_radius_px=inflation_radius_px)

    H, W = obstacle_mask.shape
    origin_xy = make_origin_xy(H, W, resolution, origin_mode=origin_mode)

    sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=max_dist)

    xs, ys, history = chompish_optimize_xy(
        p0, pN, sdf,
        origin_xy=origin_xy, resolution=resolution,
        N=N, iters=iters,
        alpha=alpha, w_obs=w_obs,
        robot_radius_m=robot_radius_m,
        safe_dist_m=safe_dist_m,
        grad_clip=grad_clip,
        keep_history=True
    )

    thetas = headings_from_path(xs, ys)
    # (thetas is what you'd use as a diff-drive heading reference)

    # We don't have polygon outline when loading from PNG, so pass None
    plot_result(sdf, origin_xy, resolution, None, p0, pN, xs, ys, history, show_checkpoints=True)

    if do_animation:
        animate_history(sdf, origin_xy, resolution, None, p0, pN, history)

if __name__ == "__main__":
    main()

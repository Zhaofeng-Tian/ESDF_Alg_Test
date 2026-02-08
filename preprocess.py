# preprocess.py
from __future__ import annotations
import numpy as np
import cv2

def load_occupancy_from_png(
    png_path: str,
    *,
    obstacle_is_dark: bool = True,
    thresh: int = 200,
) -> np.ndarray:
    """
    Load a PNG and return obstacle_mask (H,W) bool.
    - obstacle_is_dark=True: dark pixels => obstacle (common for maps)
    - thresh: grayscale threshold
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {png_path}")

    if obstacle_is_dark:
        obstacle_mask = img < thresh
    else:
        obstacle_mask = img > thresh
    return obstacle_mask

def inflate_obstacles(
    obstacle_mask: np.ndarray,
    *,
    inflation_radius_px: int = 0,
) -> np.ndarray:
    """
    Inflate obstacles by radius in pixels (Minkowski sum with a disk).
    Useful to account for robot radius before ESDF.
    """
    if inflation_radius_px <= 0:
        return obstacle_mask

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * inflation_radius_px + 1, 2 * inflation_radius_px + 1),
    )
    inflated = cv2.dilate(obstacle_mask.astype(np.uint8), k, iterations=1) > 0
    return inflated

def make_origin_xy(
    H: int, W: int, resolution: float, origin_mode: str = "center"
) -> tuple[float, float]:
    """
    Return origin_xy = (min_x, min_y) for mapping grid cells to world coords.
    - center: map is centered at (0,0)
    - zero: map lower-left corner at (0,0)
    """
    if origin_mode == "zero":
        return (0.0, 0.0)
    if origin_mode == "center":
        return (-0.5 * W * resolution, -0.5 * H * resolution)
    raise ValueError(f"Unknown origin_mode: {origin_mode}")

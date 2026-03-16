import argparse
from pathlib import Path

import cv2
import numpy as np


def load_obstacle_mask_from_png(image_path, obstacle_is_dark=True, thresh=200):
    """
    Load a PNG map as a binary obstacle mask.

    Returns
    -------
    obstacle_mask : (H, W) bool
        True means obstacle.
    gray : (H, W) uint8
        Grayscale image.
    """
    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if obstacle_is_dark:
        obstacle_mask = gray < thresh
    else:
        obstacle_mask = gray > thresh

    return obstacle_mask, gray


def dilate_obstacle_mask_px(obstacle_mask_bool, radius_px):
    """
    Dilate a binary obstacle mask by a pixel radius.
    """
    if radius_px <= 0:
        return obstacle_mask_bool.copy()

    r = int(np.ceil(radius_px))
    ksize = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_u8 = obstacle_mask_bool.astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    return dilated > 0


def save_mask_png(mask_bool, output_path):
    """
    Save obstacle mask as an image where obstacle=black, free=white.
    """
    output_path = Path(output_path)
    mask_u8 = np.where(mask_bool, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), mask_u8)
    if not ok:
        raise IOError(f"Failed to save image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dilate a map image by a pixel radius and save the dilated result."
    )
    parser.add_argument("--input", required=True, help="Input PNG map path")
    parser.add_argument("--output", required=True, help="Output PNG for dilated map")
    parser.add_argument(
        "--radius-px",
        type=float,
        default=4.0,
        help="Dilation radius in pixels",
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=200,
        help="Threshold for obstacle extraction",
    )
    parser.add_argument(
        "--obstacle-is-dark",
        action="store_true",
        help="Use dark pixels as obstacles",
    )
    args = parser.parse_args()

    obstacle_mask, _ = load_obstacle_mask_from_png(
        args.input,
        obstacle_is_dark=args.obstacle_is_dark,
        thresh=args.thresh,
    )
    dilated_mask = dilate_obstacle_mask_px(obstacle_mask, args.radius_px)
    save_mask_png(dilated_mask, args.output)

    print(f"Input map: {args.input}")
    print(f"Dilated map saved to: {args.output}")
    print(f"Dilation radius: {args.radius_px} px")


if __name__ == "__main__":
    main()
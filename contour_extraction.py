import argparse
from pathlib import Path

import cv2
import numpy as np


def load_obstacle_mask_from_png(image_path, obstacle_is_dark=True, thresh=200):
    """
    Load a PNG map as a binary obstacle mask.

    Parameters
    ----------
    image_path : str or Path
        Path to grayscale/RGB map image.
    obstacle_is_dark : bool
        True if black/dark pixels represent obstacles.
    thresh : int
        Threshold in [0, 255].

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

    Parameters
    ----------
    obstacle_mask_bool : (H, W) bool
        True means obstacle.
    radius_px : int or float
        Dilation radius in pixels.

    Returns
    -------
    dilated_mask : (H, W) bool
        Dilated obstacle mask.
    """
    if radius_px <= 0:
        return obstacle_mask_bool.copy()

    r = int(np.ceil(radius_px))
    ksize = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_u8 = obstacle_mask_bool.astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    return dilated > 0


def extract_all_contours_px(
    obstacle_mask_bool,
    min_area_px2=20.0,
    retrieval_mode="external",
):
    """
    Extract all contours from a boolean obstacle mask.
    """
    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }
    if retrieval_mode not in mode_map:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    mask_u8 = obstacle_mask_bool.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        mask_u8,
        mode_map[retrieval_mode],
        cv2.CHAIN_APPROX_NONE,
    )

    contours_out = []
    for c in contours:
        c32 = c.astype(np.float32)
        area = cv2.contourArea(c32)
        if area >= min_area_px2:
            contours_out.append(c32)

    contours_out.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return contours_out, hierarchy


def resample_closed_contour_uniform_px(contour_px, step_px):
    """
    Uniformly resample a closed contour in pixel space.
    """
    pts = contour_px.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        raise ValueError("Contour needs >= 3 points.")
    if step_px <= 0:
        raise ValueError("step_px must be > 0")

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    pts0 = pts[:-1][keep]
    pts1 = pts[1:][keep]
    seg_len = seg_len[keep]

    if seg_len.size == 0:
        raise RuntimeError("Degenerate contour with zero-length segments.")

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = cum[-1]
    n_samples = max(8, int(np.floor(perimeter / step_px)))
    s_vals = np.linspace(0.0, perimeter, n_samples, endpoint=False)

    out = np.zeros((n_samples, 2), dtype=float)
    j = 0
    for i, s in enumerate(s_vals):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        ds = s - cum[j]
        t = ds / max(seg_len[j], 1e-9)
        out[i] = (1.0 - t) * pts0[j] + t * pts1[j]

    return out


def prepare_all_contours_from_mask(
    obstacle_mask_bool,
    min_area_px2=20.0,
    retrieval_mode="external",
    sample_step_px=None,
    close_loop=True,
):
    """
    Extract all contours from a mask and optionally resample them.
    """
    contours_px, hierarchy = extract_all_contours_px(
        obstacle_mask_bool,
        min_area_px2=min_area_px2,
        retrieval_mode=retrieval_mode,
    )

    results = []
    for cid, contour_px in enumerate(contours_px):
        item = {
            "id": cid,
            "contour_px": contour_px,
            "area_px2": float(cv2.contourArea(contour_px.astype(np.float32))),
        }

        if sample_step_px is not None:
            traj_px = resample_closed_contour_uniform_px(contour_px, sample_step_px)
            if close_loop:
                traj_px = np.vstack([traj_px, traj_px[0]])
            item["traj_px"] = traj_px

        results.append(item)

    return results, hierarchy


def save_mask_png(mask_bool, output_path):
    output_path = Path(output_path)
    mask_u8 = np.where(mask_bool, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), mask_u8)
    if not ok:
        raise IOError(f"Failed to save image: {output_path}")


def save_contours_overlay(base_gray, contours_info, output_path, draw_ids=True):
    """
    Save an overlay image showing all contours and their IDs.
    """
    canvas = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    palette = [
        (255, 0, 0),
        (0, 180, 0),
        (0, 0, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 180, 180),
        (180, 0, 180),
        (80, 80, 255),
    ]

    for item in contours_info:
        cid = item["id"]
        contour = item["contour_px"].astype(np.int32)
        color = palette[cid % len(palette)]
        cv2.drawContours(canvas, [contour], -1, color, 1)

        if draw_ids:
            pts = contour.reshape(-1, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(
                canvas,
                f"{cid}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    output_path = Path(output_path)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise IOError(f"Failed to save image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dilate a map image by pixels, then extract all contours from the dilated image."
    )
    parser.add_argument("--input", required=True, help="Input PNG map path")
    parser.add_argument(
        "--dilated-output",
        default="dilated_map.png",
        help="Output path for dilated binary image",
    )
    parser.add_argument(
        "--overlay-output",
        default="dilated_contours_overlay.png",
        help="Output path for contour overlay image",
    )
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
    parser.add_argument(
        "--retrieval-mode",
        choices=["external", "tree", "ccomp"],
        default="external",
        help="Contour retrieval mode",
    )
    parser.add_argument(
        "--min-area-px2",
        type=float,
        default=20.0,
        help="Minimum contour area to keep",
    )
    parser.add_argument(
        "--sample-step-px",
        type=float,
        default=None,
        help="Optional uniform contour resampling step in pixels",
    )
    args = parser.parse_args()

    obstacle_mask, gray = load_obstacle_mask_from_png(
        args.input,
        obstacle_is_dark=args.obstacle_is_dark,
        thresh=args.thresh,
    )

    dilated_mask = dilate_obstacle_mask_px(obstacle_mask, args.radius_px)
    save_mask_png(dilated_mask, args.dilated_output)

    contours_info, hierarchy = prepare_all_contours_from_mask(
        dilated_mask,
        min_area_px2=args.min_area_px2,
        retrieval_mode=args.retrieval_mode,
        sample_step_px=args.sample_step_px,
        close_loop=True,
    )

    dilated_gray = np.where(dilated_mask, 0, 255).astype(np.uint8)
    save_contours_overlay(dilated_gray, contours_info, args.overlay_output, draw_ids=True)

    print(f"Input map: {args.input}")
    print(f"Dilated map saved to: {args.dilated_output}")
    print(f"Contour overlay saved to: {args.overlay_output}")
    print(f"Number of contours found: {len(contours_info)}")
    for item in contours_info:
        msg = f"  contour {item['id']}: area={item['area_px2']:.1f}px^2, raw_pts={len(item['contour_px'])}"
        if "traj_px" in item:
            msg += f", sampled_pts={len(item['traj_px'])}"
        print(msg)

    if hierarchy is not None:
        print(f"Hierarchy shape: {hierarchy.shape}")


if __name__ == "__main__":
    main()
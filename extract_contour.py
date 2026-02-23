from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for contour extraction. Install with: pip install opencv-python"
        ) from exc
    return cv2


def _require_pyclipper():
    try:
        import pyclipper  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyclipper is required for contour dilation. Install with: pip install pyclipper"
        ) from exc
    return pyclipper


def load_obstacle_mask(image_path: str, obstacle_is_dark: bool | None = None) -> np.ndarray:
    """Load grayscale image and return obstacle mask as uint8 {0,255}."""
    cv2 = _require_cv2()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_mask = otsu == 0
    light_mask = otsu == 255

    if obstacle_is_dark is None:
        obstacle_mask = dark_mask if dark_mask.sum() <= light_mask.sum() else light_mask
    else:
        obstacle_mask = dark_mask if obstacle_is_dark else light_mask

    mask_u8 = obstacle_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8


def extract_largest_contour(mask_u8: np.ndarray, min_area: float = 20.0) -> np.ndarray:
    """Extract largest external contour from binary obstacle mask."""
    cv2 = _require_cv2()

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour found in obstacle mask.")

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    largest = contours[idx]
    if areas[idx] < min_area:
        raise RuntimeError(f"Largest contour area too small: {areas[idx]:.3f}")
    return largest


def contour_to_line_segments(contour: np.ndarray, closed: bool = True) -> List[Segment]:
    """Convert contour points into piecewise-linear segments."""
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 2:
        return []

    segments: List[Segment] = []
    for i in range(len(pts) - 1):
        p0 = (float(pts[i, 0]), float(pts[i, 1]))
        p1 = (float(pts[i + 1, 0]), float(pts[i + 1, 1]))
        segments.append((p0, p1))

    if closed and len(pts) > 2:
        p_last = (float(pts[-1, 0]), float(pts[-1, 1]))
        p_first = (float(pts[0, 0]), float(pts[0, 1]))
        segments.append((p_last, p_first))

    return segments


def dilate_contour_pyclipper(
    contour: np.ndarray,
    radius_px: float,
    join_type: str = "round",
    arc_tolerance: float = 0.25,
) -> np.ndarray:
    """Dilate (offset outward) contour using pyclipper by radius in pixels."""
    if radius_px <= 0:
        return contour.copy()

    pyclipper = _require_pyclipper()

    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 contour points for dilation.")

    join_map = {
        "round": pyclipper.JT_ROUND,
        "miter": pyclipper.JT_MITER,
        "square": pyclipper.JT_SQUARE,
    }
    if join_type not in join_map:
        raise ValueError(f"Unsupported join_type: {join_type}")

    scale = 1000.0
    path = [(int(round(p[0] * scale)), int(round(p[1] * scale))) for p in pts]

    co = pyclipper.PyclipperOffset()
    co.ArcTolerance = float(arc_tolerance) * scale
    co.AddPath(path, join_map[join_type], pyclipper.ET_CLOSEDPOLYGON)
    solution = co.Execute(float(radius_px) * scale)
    if not solution:
        raise RuntimeError("pyclipper offset produced no output contour.")

    best = max(solution, key=lambda s: abs(pyclipper.Area(s)))
    best_np = np.asarray(best, dtype=float) / scale
    return best_np.reshape(-1, 1, 2)


def save_segments_json(segments: List[Segment], out_path: str) -> None:
    data = [
        {
            "p0": [seg[0][0], seg[0][1]],
            "p1": [seg[1][0], seg[1][1]],
        }
        for seg in segments
    ]
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_contours_and_segments(
    image_path: str,
    mask_u8: np.ndarray,
    contour: np.ndarray,
    segments: List[Segment],
    dilated_contour: np.ndarray | None = None,
    dilated_segments: List[Segment] | None = None,
) -> None:
    cv2 = _require_cv2()

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask_u8, cmap="gray")
    axes[1].set_title("Extracted + Dilated Contours")
    axes[1].axis("equal")

    if len(contour) > 0:
        pts = contour.reshape(-1, 2)
        axes[1].plot(pts[:, 0], pts[:, 1], "r-", linewidth=1.5, label="Original contour")

    if segments:
        lc = LineCollection(
            segments, colors="cyan", linewidths=0.6, alpha=0.8, label="Original segments"
        )
        axes[1].add_collection(lc)

    if dilated_contour is not None and len(dilated_contour) > 0:
        dpts = dilated_contour.reshape(-1, 2)
        axes[1].plot(
            dpts[:, 0],
            dpts[:, 1],
            "y-",
            linewidth=1.4,
            label="Dilated contour",
        )

    if dilated_segments:
        d_lc = LineCollection(
            dilated_segments,
            colors="lime",
            linewidths=0.6,
            alpha=0.8,
            label="Dilated segments",
        )
        axes[1].add_collection(d_lc)

    axes[1].invert_yaxis()
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract obstacle contour, optional pyclipper dilation, and return line segments."
    )
    parser.add_argument(
        "--image", type=str, default="circle_obstacle.png", help="Path to obstacle image."
    )
    parser.add_argument(
        "--obstacle-is-dark", action="store_true", help="Set if obstacle pixels are dark."
    )
    parser.add_argument(
        "--obstacle-is-light", action="store_true", help="Set if obstacle pixels are light."
    )
    parser.add_argument("--min-area", type=float, default=20.0, help="Minimum contour area.")
    parser.add_argument(
        "--dilate-radius",
        type=float,
        default=0.0,
        help="Dilate contour outward by this radius in pixels (pyclipper).",
    )
    parser.add_argument(
        "--offset-join",
        type=str,
        default="round",
        choices=["round", "miter", "square"],
        help="pyclipper join style for contour offset.",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib display.")
    parser.add_argument(
        "--segments-out",
        type=str,
        default="",
        help="Optional JSON output path for original line segments.",
    )
    parser.add_argument(
        "--dilated-segments-out",
        type=str,
        default="",
        help="Optional JSON output path for dilated line segments.",
    )
    args = parser.parse_args()

    if args.obstacle_is_dark and args.obstacle_is_light:
        raise ValueError("Choose only one of --obstacle-is-dark or --obstacle-is-light")

    mode = None
    if args.obstacle_is_dark:
        mode = True
    elif args.obstacle_is_light:
        mode = False

    image_path = str(Path(args.image))
    mask_u8 = load_obstacle_mask(image_path, obstacle_is_dark=mode)
    contour = extract_largest_contour(mask_u8, min_area=args.min_area)
    segments = contour_to_line_segments(contour, closed=True)

    dilated_contour = None
    dilated_segments: List[Segment] = []
    if args.dilate_radius > 0:
        dilated_contour = dilate_contour_pyclipper(
            contour,
            radius_px=args.dilate_radius,
            join_type=args.offset_join,
        )
        dilated_segments = contour_to_line_segments(dilated_contour, closed=True)

    print(f"Original contour points: {len(contour)}")
    print(f"Original line segments: {len(segments)}")
    print("First 10 original segments (x1, y1) -> (x2, y2):")
    for i, seg in enumerate(segments[:10]):
        print(f"  {i:02d}: {seg[0]} -> {seg[1]}")

    if dilated_contour is not None:
        print(f"Dilated contour points: {len(dilated_contour)}")
        print(f"Dilated line segments: {len(dilated_segments)}")
        print("First 10 dilated segments (x1, y1) -> (x2, y2):")
        for i, seg in enumerate(dilated_segments[:10]):
            print(f"  {i:02d}: {seg[0]} -> {seg[1]}")

    if args.segments_out:
        save_segments_json(segments, args.segments_out)
        print(f"Saved original segments to: {args.segments_out}")

    if args.dilated_segments_out and dilated_segments:
        save_segments_json(dilated_segments, args.dilated_segments_out)
        print(f"Saved dilated segments to: {args.dilated_segments_out}")

    if not args.no_show:
        plot_contours_and_segments(
            image_path,
            mask_u8,
            contour,
            segments,
            dilated_contour=dilated_contour,
            dilated_segments=dilated_segments,
        )


if __name__ == "__main__":
    main()

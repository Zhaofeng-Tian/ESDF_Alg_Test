import os
import numpy as np
import cv2


# =========================
# ====== PARAMETERS ======
# =========================

# Map size in world units (meters)
MAP_WIDTH_M = 10.0
MAP_HEIGHT_M = 10.0

# Resolution (meters per pixel)
RESOLUTION = 0.05

# Output
OUTPUT_DIR = "./"
OUTPUT_NAME = "l_shape_obstacle_45deg.png"

# Pixel values
FREE_VALUE = 255
OCC_VALUE = 0

# Obstacle catalog:
# - circle: {"type": "circle", "cx": x, "cy": y, "r": radius}
# - rectangle: {"type": "rectangle", "x": x, "y": y, "w": width, "h": height}
# - l_shape: {
#       "type": "l_shape",
#       "x": corner_x, "y": corner_y,
#       "w": overall_width, "h": overall_height,
#       "thickness": arm_thickness,
#       "orientation": "ll" | "lr" | "ul" | "ur"
#   }
# Corner means:
# - ll: lower-left corner of outer bounding box
# - lr: lower-right
# - ul: upper-left
# - ur: upper-right
OBSTACLES = [
    {
        "type": "l_shape",
        "x": 3.0,
        "y": 3.0,
        "w": 3.0,
        "h": 3.0,
        "thickness": 0.6,
        "orientation": "ll",
        "vertical_angle_deg": 45.0,
    },
]


def world_to_px(v_m):
    return int(round(v_m / RESOLUTION))


def draw_circle(img, obs):
    cx = world_to_px(obs["cx"])
    cy = world_to_px(obs["cy"])
    r = max(1, world_to_px(obs["r"]))
    cv2.circle(img, center=(cx, cy), radius=r, color=OCC_VALUE, thickness=-1)


def draw_rectangle(img, obs):
    x0 = world_to_px(obs["x"])
    y0 = world_to_px(obs["y"])
    w = max(1, world_to_px(obs["w"]))
    h = max(1, world_to_px(obs["h"]))
    x1 = x0 + w
    y1 = y0 + h
    cv2.rectangle(img, (x0, y0), (x1, y1), color=OCC_VALUE, thickness=-1)


def draw_l_shape(img, obs):
    x = obs["x"]
    y = obs["y"]
    w = obs["w"]
    h = obs["h"]
    t = obs["thickness"]
    ori = obs.get("orientation", "ll").lower()
    vertical_angle_deg = float(obs.get("vertical_angle_deg", 90.0))

    if t <= 0.0 or t > min(w, h):
        raise ValueError("l_shape thickness must satisfy 0 < thickness <= min(w, h)")

    if ori not in {"ll", "lr", "ul", "ur"}:
        raise ValueError("l_shape orientation must be one of: ll, lr, ul, ur")

    if ori == "ll":
        # Horizontal arm (kept axis-aligned)
        rects = [{"x": x, "y": y, "w": w, "h": t}]

        # Vertical arm can be rotated around the corner (x, y)
        # angle is measured from +x axis in image coordinates.
        x0 = world_to_px(x)
        y0 = world_to_px(y)
        len_px = max(1.0, h / RESOLUTION)
        thick_px = max(1.0, t / RESOLUTION)
        theta = np.deg2rad(vertical_angle_deg)
        u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        n = np.array([-u[1], u[0]], dtype=float)
        p0 = np.array([x0, y0], dtype=float)
        p1 = p0 + len_px * u
        half_t = 0.5 * thick_px
        poly = np.array(
            [
                p0 + half_t * n,
                p0 - half_t * n,
                p1 - half_t * n,
                p1 + half_t * n,
            ],
            dtype=np.float32,
        )
        cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), OCC_VALUE)
    elif ori == "lr":
        rects = [
            {"x": x - w, "y": y, "w": w, "h": t},
            {"x": x - t, "y": y, "w": t, "h": h},
        ]
    elif ori == "ul":
        rects = [
            {"x": x, "y": y - t, "w": w, "h": t},
            {"x": x, "y": y - h, "w": t, "h": h},
        ]
    else:  # ur
        rects = [
            {"x": x - w, "y": y - t, "w": w, "h": t},
            {"x": x - t, "y": y - h, "w": t, "h": h},
        ]

    for r in rects:
        draw_rectangle(img, r)


def draw_obstacle(img, obs):
    obs_type = obs.get("type", "").lower()
    if obs_type == "circle":
        draw_circle(img, obs)
    elif obs_type == "rectangle":
        draw_rectangle(img, obs)
    elif obs_type == "l_shape":
        draw_l_shape(img, obs)
    else:
        raise ValueError(f"Unsupported obstacle type: {obs_type}")


def main():
    width_px = world_to_px(MAP_WIDTH_M)
    height_px = world_to_px(MAP_HEIGHT_M)
    print(f"Map size: {width_px} x {height_px} pixels")

    img = np.full((height_px, width_px), FREE_VALUE, dtype=np.uint8)

    for i, obs in enumerate(OBSTACLES):
        draw_obstacle(img, obs)
        print(f"Drew obstacle {i}: {obs['type']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

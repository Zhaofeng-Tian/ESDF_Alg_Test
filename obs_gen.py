import os
import numpy as np
import cv2


# =========================
# ====== PARAMETERS ======
# =========================

# Map size in world units (meters)
MAP_WIDTH_M  = 10.0
MAP_HEIGHT_M = 10.0

# Resolution (meters per pixel)
RESOLUTION = 0.05   # smaller = higher resolution

# Circle obstacle (world coordinates)
CIRCLE_CENTER_X = 5.0
CIRCLE_CENTER_Y = 5.0
CIRCLE_RADIUS   = 1.0

# Output
OUTPUT_DIR  = "./"        # project folder
OUTPUT_NAME = "circle_obstacle.png"

# Pixel values
FREE_VALUE = 255          # white
OCC_VALUE  = 0            # black


# =========================
# ====== MAIN LOGIC ======
# =========================

def main():
    # Compute grid size
    width_px  = int(MAP_WIDTH_M  / RESOLUTION)
    height_px = int(MAP_HEIGHT_M / RESOLUTION)

    print(f"Map size: {width_px} x {height_px} pixels")

    # Create empty map
    img = np.full((height_px, width_px), FREE_VALUE, dtype=np.uint8)

    # Convert world → pixel
    cx_px = int(CIRCLE_CENTER_X / RESOLUTION)
    cy_px = int(CIRCLE_CENTER_Y / RESOLUTION)
    r_px  = int(CIRCLE_RADIUS / RESOLUTION)

    # Draw filled circle
    cv2.circle(
        img,
        center=(cx_px, cy_px),
        radius=r_px,
        color=OCC_VALUE,
        thickness=-1,
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    cv2.imwrite(out_path, img)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

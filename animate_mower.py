import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d
from mower_model import MowerFrontModel


# =========================================================
# USER SETTINGS
# =========================================================
EXPORT_MODE = "mov"      # "gif" or "mov"
OUTPUT_NAME = "mower_animation"
FPS = 30
MAX_DELTA_DEG = 10.0


# =========================================================
# Utility
# =========================================================
def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def refine_turns(traj_xy, beta, max_delta_deg):
    max_delta = np.deg2rad(max_delta_deg)

    new_xy = [traj_xy[0]]
    new_beta = [beta[0]]

    for k in range(len(beta) - 1):
        p0 = traj_xy[k]
        p1 = traj_xy[k + 1]

        b0 = beta[k]
        b1 = beta[k + 1]

        d_beta = wrap_to_pi(b1 - b0)
        abs_d = abs(d_beta)

        n_seg = max(1, int(np.ceil(abs_d / max_delta)))

        for i in range(1, n_seg + 1):
            t = i / n_seg

            p_interp = (1 - t) * p0 + t * p1
            b_interp = wrap_to_pi(b0 + t * d_beta)

            new_xy.append(p_interp)
            new_beta.append(b_interp)

    return np.array(new_xy), np.array(new_beta)


# =========================================================
# Load optimized trajectory
# =========================================================
data = np.load("optimized_traj.npz")

traj_xy = data["traj_xy"]
beta = data["beta"]
resolution = float(data["resolution"])

traj_xy, beta = refine_turns(traj_xy, beta, MAX_DELTA_DEG)
print("Refined trajectory length:", len(traj_xy))


# =========================================================
# Recompute ESDF
# =========================================================
obstacle_mask = load_occupancy_from_png(
    "l_shape_obstacle_45deg.png",
    obstacle_is_dark=True,
    thresh=200,
)

sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=5.0)
robot = MowerFrontModel(circle_radius=0.05)


# =========================================================
# Visualization Setup
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8))

sdf_vis = sdf.copy()
sdf_vis[obstacle_mask] = -1.0

ax.imshow(
    sdf_vis,
    cmap="RdYlGn",
    vmin=-1.0,
    vmax=4.0,
    origin="lower",
)

ax.plot(
    traj_xy[:, 0] / resolution,
    traj_xy[:, 1] / resolution,
    "k--",
    linewidth=1.0,
)

ax.set_title("Mower Animation")
ax.axis("off")

body_patches = []
heading_arrow = None


# =========================================================
# Animation Update
# =========================================================
def update(frame):
    global body_patches, heading_arrow

    # remove previous primitives
    for p in body_patches:
        p.remove()
    body_patches = []

    if heading_arrow is not None:
        heading_arrow.remove()

    state = (
        float(traj_xy[frame, 0]),
        float(traj_xy[frame, 1]),
        float(beta[frame]),
    )

    prims = robot.world_centers_and_jacobians(state)

    for pr in prims:
        c = pr["center_w"]
        r = pr["radius"]
        tag = pr["tag"]

        color = "cyan" if tag == "collision" else "magenta"

        circ = Circle(
            (c[0] / resolution, c[1] / resolution),
            radius=r / resolution,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )

        ax.add_patch(circ)
        body_patches.append(circ)

    # heading arrow
    arrow_scale = 0.2
    heading_arrow = ax.arrow(
        state[0] / resolution,
        state[1] / resolution,
        arrow_scale * np.cos(state[2]) / resolution,
        arrow_scale * np.sin(state[2]) / resolution,
        head_width=0.1,
        color="blue",
    )

    return body_patches


# =========================================================
# Create animation
# =========================================================
anim = FuncAnimation(
    fig,
    update,
    frames=len(traj_xy),
    interval=1000 / FPS,
    blit=False,
    repeat=False,
)


# =========================================================
# Export
# =========================================================
if EXPORT_MODE.lower() == "gif":
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=FPS)
    anim.save(f"{OUTPUT_NAME}.gif", writer=writer)
    print(f"Saved {OUTPUT_NAME}.gif")

elif EXPORT_MODE.lower() == "mov":
    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    anim.save(f"{OUTPUT_NAME}.mov", writer=writer)
    print(f"Saved {OUTPUT_NAME}.mov")

else:
    raise ValueError("EXPORT_MODE must be 'gif' or 'mov'")

plt.show()
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query


# =========================================================
# Generate horizontal line crossing upper part of circle
# =========================================================
def generate_horizontal_line_across_circle(
    cx_px: int,
    cy_px: int,
    r_px: int,
    width_px: int,
    offset_ratio: float = 0.3,
):
    offset_px = int(offset_ratio * r_px)
    y = cy_px + offset_px
    xs = np.arange(0, width_px)
    ys = np.full_like(xs, y)
    return xs, ys


# =========================================================
# Extract start/end in free space + downsample
# =========================================================
def extract_and_downsample_trajectory(xs, ys, obstacle_mask, stride_cells):

    free_mask = ~obstacle_mask[ys, xs]
    free_indices = np.where(free_mask)[0]

    if free_indices.size < 2:
        raise RuntimeError("Not enough free-space points")

    start_idx = free_indices[0]
    end_idx = free_indices[-1]

    xs_clip = xs[start_idx:end_idx + 1]
    ys_clip = ys[start_idx:end_idx + 1]

    xs_ds = xs_clip[::stride_cells]
    ys_ds = ys_clip[::stride_cells]

    if xs_ds[-1] != xs_clip[-1] or ys_ds[-1] != ys_clip[-1]:
        xs_ds = np.append(xs_ds, xs_clip[-1])
        ys_ds = np.append(ys_ds, ys_clip[-1])

    return xs_clip, ys_clip, xs_ds, ys_ds


# =========================================================
# CHOMP obstacle hinge cost
# =========================================================
def chomp_obstacle_cost_and_grad(d_vals, grads, epsilon):

    cost = np.zeros_like(d_vals)
    grad_cost = np.zeros_like(grads)

    mask = d_vals < epsilon

    cost[mask] = 0.5 * (d_vals[mask] - epsilon) ** 2
    grad_cost[mask] = ((d_vals[mask] - epsilon)[:, None] * grads[mask])

    return cost, grad_cost


# =========================================================
# MAIN
# =========================================================
def main():

    image_path = "circle_obstacle.png"
    resolution = 0.05
    max_dist = 5.0
    circle_radius_m = 1.0

    epsilon = 0.2  # CHOMP safety margin (meters)

    arrow_scale = 1.0

    obstacle_mask = load_occupancy_from_png(
        image_path,
        obstacle_is_dark=True,
        thresh=200,
    )

    H, W = obstacle_mask.shape
    print("[INFO] Map shape:", (H, W))

    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=max_dist,
    )

    print("[INFO] ESDF range:", sdf.min(), sdf.max())

    # ---- Generate line
    cx_px = W // 2
    cy_px = H // 2
    r_px = int(circle_radius_m / resolution)

    xs, ys = generate_horizontal_line_across_circle(
        cx_px, cy_px, r_px, W
    )

    # ---- Downsample every 50 cm
    stride_cells = 10
    xs_clip, ys_clip, xs_traj, ys_traj = extract_and_downsample_trajectory(
        xs, ys, obstacle_mask, stride_cells
    )

    print("[INFO] Trajectory states:", len(xs_traj))

    # ---- Continuous ESDF query
    origin_xy = (0.0, 0.0)

    grads = []
    d_vals = []

    for x_px, y_px in zip(xs_traj, ys_traj):

        xw = x_px * resolution
        yw = y_px * resolution

        d, grad = sdf_query(
            sdf,
            xw,
            yw,
            origin_xy=origin_xy,
            resolution=resolution,
        )

        grads.append(grad)
        d_vals.append(d)

    grads = np.array(grads)
    d_vals = np.array(d_vals)

    print("[INFO] Gradient norm min/max:",
          np.linalg.norm(grads, axis=1).min(),
          np.linalg.norm(grads, axis=1).max())

    # ---- CHOMP obstacle cost
    cost_vals, grad_cost = chomp_obstacle_cost_and_grad(
        d_vals, grads, epsilon
    )

    grad_cost = - grad_cost  # Negate to get descent direction
    print("[INFO] Cost gradient norms:",
          np.linalg.norm(grad_cost, axis=1))

    # ---- Convert to pixel space for plotting
    grads_px = grads / resolution
    grad_cost_px = grad_cost / resolution

    print("grads", grads)
    print("grad_cost", grad_cost)   

    # =========================================================
    # Visualization
    # =========================================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Obstacle plot
    axs[0].imshow(obstacle_mask, cmap="gray", origin="lower")
    axs[0].plot(xs_clip, ys_clip, color="cyan", linewidth=1, alpha=0.6)
    axs[0].scatter(xs_traj, ys_traj, s=60, color="blue", zorder=3)

    axs[0].scatter(
        [xs_traj[0], xs_traj[-1]],
        [ys_traj[0], ys_traj[-1]],
        s=120,
        color="yellow",
        edgecolors="black",
        zorder=4,
    )

    # ESDF gradient (magenta)
    axs[0].quiver(
        xs_traj, ys_traj,
        grads_px[:, 0], grads_px[:, 1],
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color="orange",
        width=0.004,
    )

    # CHOMP cost gradient (red)
    axs[0].quiver(
        xs_traj, ys_traj,
        grad_cost_px[:, 0], grad_cost_px[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.5*arrow_scale,
        color="red",
        width=0.002,
    )

    axs[0].set_title("Obstacle + ∇SDF (magenta) + ∇Cost (red)")
    axs[0].axis("off")

    # ---- ESDF heatmap
    sdf_vis = sdf.copy()
    sdf_vis[obstacle_mask] = -1.0

    im = axs[1].imshow(
        sdf_vis,
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=4.0,
        origin="lower",
    )

    axs[1].plot(xs_clip, ys_clip, color="cyan", linewidth=1, alpha=0.6)
    axs[1].scatter(xs_traj, ys_traj, s=60, color="blue", zorder=3)

    axs[1].quiver(
        xs_traj, ys_traj,
        grads_px[:, 0], grads_px[:, 1],
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color="orange",
        width=0.004,
    )

    axs[1].quiver(
        xs_traj, ys_traj,
        grad_cost_px[:, 0], grad_cost_px[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.5*arrow_scale,
        color="red",
        width=0.002,
    )

    axs[1].set_title("Signed ESDF + Gradients")
    axs[1].axis("off")

    plt.colorbar(im, ax=axs[1], fraction=0.046)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

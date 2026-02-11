import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query


# =========================================================
# Line generation
# =========================================================
def generate_horizontal_line(cx_px, cy_px, r_px, width_px, offset_ratio=0.3):
    y = cy_px + int(offset_ratio * r_px)
    xs = np.arange(0, width_px)
    ys = np.full_like(xs, y)
    return xs, ys


# =========================================================
# Downsample
# =========================================================
def extract_and_downsample(xs, ys, obstacle_mask, stride):
    free_mask = ~obstacle_mask[ys, xs]
    free_idx = np.where(free_mask)[0]

    start = free_idx[0]
    end = free_idx[-1]

    xs_clip = xs[start:end+1]
    ys_clip = ys[start:end+1]

    xs_ds = xs_clip[::stride]
    ys_ds = ys_clip[::stride]

    if xs_ds[-1] != xs_clip[-1]:
        xs_ds = np.append(xs_ds, xs_clip[-1])
        ys_ds = np.append(ys_ds, ys_clip[-1])

    return xs_ds, ys_ds


# =========================================================
# CHOMP hinge obstacle cost
# =========================================================
def obstacle_cost_grad(d_vals, grads, epsilon):
    grad_cost = np.zeros_like(grads)
    mask = d_vals < epsilon
    grad_cost[mask] = ((d_vals[mask] - epsilon)[:, None] * grads[mask])
    return grad_cost


# =========================================================
# Smoothness (discrete Laplacian)
# =========================================================
def smoothness_grad(traj):
    grad = np.zeros_like(traj)
    grad[1:-1] = 2*traj[1:-1] - traj[0:-2] - traj[2:]
    return grad


# =========================================================
# MAIN
# =========================================================
def main():

    resolution = 0.05
    epsilon = 0.2
    step_size = 0.1
    lambda_smooth = 1.0
    iters = 60

    obstacle_mask = load_occupancy_from_png(
        "circle_obstacle.png",
        obstacle_is_dark=True,
        thresh=200,
    )

    H, W = obstacle_mask.shape

    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=5.0,
    )

    cx = W // 2
    cy = H // 2
    r_px = int(1.0 / resolution)

    xs, ys = generate_horizontal_line(cx, cy, r_px, W)
    xs_traj, ys_traj = extract_and_downsample(xs, ys, obstacle_mask, stride=10)

    traj_world = np.stack(
        [xs_traj * resolution, ys_traj * resolution],
        axis=1
    )

    traj_initial = traj_world.copy()
    traj_history = [traj_world.copy()]

    origin_xy = (0.0, 0.0)

    # =========================================================
    # Optimization loop
    # =========================================================
    for k in range(iters):

        grads = []
        d_vals = []

        for x, y in traj_world:
            d, g = sdf_query(
                sdf, x, y,
                origin_xy=origin_xy,
                resolution=resolution
            )
            grads.append(g)
            d_vals.append(d)

        grads = np.array(grads)
        d_vals = np.array(d_vals)

        grad_obs = obstacle_cost_grad(d_vals, grads, epsilon)
        grad_smooth = smoothness_grad(traj_world)

        total_grad = grad_obs + lambda_smooth * grad_smooth

        total_grad[0] = 0.0
        total_grad[-1] = 0.0

        traj_world -= step_size * total_grad

        if (k + 1) % 5 == 0:
            traj_history.append(traj_world.copy())

        if k % 10 == 0:
            print(f"Iter {k}, min dist: {d_vals.min():.3f}")

    # =========================================================
    # Visualization
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

    # --- Initial trajectory (very clear)
    ax.plot(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Initial"
    )

    ax.scatter(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        color="black",
        s=70,
        zorder=5
    )

    # --- Intermediate trajectories
    for i, traj in enumerate(traj_history[1:-1]):

        alpha = 0.3 + 0.5 * (i / len(traj_history))

        ax.plot(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="orange",
            alpha=alpha,
            linewidth=1.5,
        )

        ax.scatter(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="orange",
            s=40,
            alpha=alpha,
        )

    # --- Final trajectory
    traj_final = traj_history[-1]

    ax.plot(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        linewidth=3,
        label="Final"
    )

    ax.scatter(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        s=80,
        zorder=6
    )

    ax.set_title("CHOMP Optimization Progress (Every 5 Iterations)")
    ax.axis("off")
    ax.legend()

    plt.show()



if __name__ == "__main__":
    main()

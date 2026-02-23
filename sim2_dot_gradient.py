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


def compute_terms(traj_world, sdf, origin_xy, resolution, epsilon):
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
    return d_vals, grad_obs, grad_smooth

def total_cost(traj_world, sdf, origin_xy, resolution, epsilon, lambda_smooth):
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

    # obstacle cost
    obs_cost = 0.5 * np.sum((d_vals[d_vals < epsilon] - epsilon)**2)

    # smoothness cost
    smooth_cost = np.sum(
        np.linalg.norm(traj_world[1:] - traj_world[:-1], axis=1)**2
    )

    return obs_cost + lambda_smooth * smooth_cost


# =========================================================
# MAIN
# =========================================================
def main():

    resolution = 0.05
    epsilon = 0.2
    step_size = 1
    lambda_smooth = 0.05
    iters = 50

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

    _, grad_obs0, grad_smooth0 = compute_terms(
        traj_world, sdf, origin_xy, resolution, epsilon
    )
    grad_obs_history = [grad_obs0.copy()]
    grad_smooth_history = [grad_smooth0.copy()]

    # # =========================================================
    # # Optimization loop
    # # =========================================================
    # for k in range(iters):

    #     d_vals, grad_obs, grad_smooth = compute_terms(
    #         traj_world, sdf, origin_xy, resolution, epsilon
    #     )

    #     total_grad = grad_obs + lambda_smooth * grad_smooth

    #     total_grad[0] = 0.0
    #     total_grad[-1] = 0.0

    #     traj_world -= step_size * total_grad

    #     if (k + 1) % 10 == 0:
    #         traj_history.append(traj_world.copy())

    #         _, grad_obs_snap, grad_smooth_snap = compute_terms(
    #             traj_world, sdf, origin_xy, resolution, epsilon
    #         )
    #         grad_obs_history.append(grad_obs_snap.copy())
    #         grad_smooth_history.append(grad_smooth_snap.copy())

    #         print("total gradient norm:", np.linalg.norm(total_grad))
    #     if k % 10 == 0:
    #         print(f"Iter {k}, min dist: {d_vals.min():.3f}")

    # =========================================================
    # Optimization loop (with line search)
    # =========================================================
    for k in range(iters):

        # ---- compute gradients
        d_vals, grad_obs, grad_smooth = compute_terms(
            traj_world, sdf, origin_xy, resolution, epsilon
        )

        total_grad = grad_obs + lambda_smooth * grad_smooth

        # fix endpoints
        total_grad[0] = 0.0
        total_grad[-1] = 0.0

        descent = -total_grad

        # ---- backtracking line search
        alpha = 1.0  # you can try 2.0 or 3.0 if you want

        c0 = total_cost(
            traj_world, sdf, origin_xy,
            resolution, epsilon, lambda_smooth
        )

        while True:

            traj_trial = traj_world + alpha * descent

            c_trial = total_cost(
                traj_trial, sdf, origin_xy,
                resolution, epsilon, lambda_smooth
            )

            if c_trial < c0:
                break

            alpha *= 0.5

            if alpha < 1e-4:
                # No improving step found; keep current trajectory.
                traj_trial = traj_world.copy()
                break

        traj_world = traj_trial

        if (k + 1) % 5 == 0:
            traj_history.append(traj_world.copy())
            _, grad_obs_snap, grad_smooth_snap = compute_terms(
                traj_world, sdf, origin_xy, resolution, epsilon
            )
            grad_obs_history.append(grad_obs_snap.copy())
            grad_smooth_history.append(grad_smooth_snap.copy())

        if k % 10 == 0:
            print(f"Iter {k}, min dist: {d_vals.min():.3f}, alpha: {alpha:.4f}")


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

    # --- Intermediate trajectories with gradient arrows
    for i, traj in enumerate(traj_history[1:-1], start=1):

        alpha = 0.3 + 0.5 * (i / len(traj_history))

        ax.plot(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            alpha=alpha,
            linewidth=1.5,
        )

        ax.scatter(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            s=40,
            alpha=alpha,
        )

        grad_obs = grad_obs_history[i]
        grad_smooth = grad_smooth_history[i]

        ax.quiver(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            grad_obs[:, 0],
            grad_obs[:, 1],
            color="green",
            angles="xy",
            scale_units="xy",
            width=0.003,
            alpha=alpha,
            label="Obstacle grad" if i == 1 else None,
        )

        ax.quiver(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            grad_smooth[:, 0],
            grad_smooth[:, 1],
            color="blue",
            angles="xy",
            scale_units="xy",
            width=0.003,
            alpha=alpha,
            label="Smoothness grad" if i == 1 else None,
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

    grad_obs_final = grad_obs_history[-1]
    grad_smooth_final = grad_smooth_history[-1]

    ax.quiver(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        grad_obs_final[:, 0],
        grad_obs_final[:, 1],
        color="green",
        angles="xy",
        scale_units="xy",
        width=0.004,
    )

    ax.quiver(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        grad_smooth_final[:, 0],
        grad_smooth_final[:, 1],
        color="blue",
        angles="xy",
        scale_units="xy",
        width=0.004,
    )

    ax.set_title("CHOMP Optimization Progress (Every 5 Iterations)")
    ax.axis("off")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()

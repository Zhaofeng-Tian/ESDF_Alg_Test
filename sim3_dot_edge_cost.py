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
# Edge-seeking + inside barrier terms
# =========================================================
def edge_target_grad(d_vals, grads, d_target):
    # Pull each point toward target clearance outside obstacle.
    return ((d_vals - d_target)[:, None] * grads)


def inside_barrier_grad(d_vals, grads, margin_inside):
    # Penalize points that go inside (or below safety margin).
    grad_cost = np.zeros_like(grads)
    mask = d_vals < margin_inside
    grad_cost[mask] = ((d_vals[mask] - margin_inside)[:, None] * grads[mask])
    return grad_cost


# =========================================================
# Smoothness (discrete Laplacian)
# =========================================================
def smoothness_grad(traj):
    grad = np.zeros_like(traj)
    grad[1:-1] = 2*traj[1:-1] - traj[0:-2] - traj[2:]
    return grad


def compute_terms(
    traj_world,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
):
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

    grad_edge = edge_target_grad(d_vals, grads, d_target)
    grad_inside = inside_barrier_grad(d_vals, grads, margin_inside)
    grad_smooth = smoothness_grad(traj_world)
    return d_vals, grad_edge, grad_inside, grad_smooth


def total_cost(
    traj_world,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
    w_edge,
    w_inside,
    lambda_smooth,
):
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

    # edge target cost: minimum at d = d_target (outside obstacle)
    edge_cost = 0.5 * np.sum((d_vals - d_target)**2)

    # inside barrier cost: active only if d < margin_inside
    inside_mask = d_vals < margin_inside
    inside_cost = 0.5 * np.sum((d_vals[inside_mask] - margin_inside)**2)

    # smoothness cost
    smooth_cost = np.sum(
        np.linalg.norm(traj_world[1:] - traj_world[:-1], axis=1)**2
    )

    return (
        w_edge * edge_cost
        + w_inside * inside_cost
        + lambda_smooth * smooth_cost
    )


# =========================================================
# MAIN
# =========================================================
def main():

    resolution = 0.05
    # ---- Tuning block
    d_target = 0.06       # desired clearance to obstacle edge (meters)
    margin_inside = 0  # keep >= 0.0 to avoid going inside obstacle
    w_edge = 1.0          # pull toward d_target
    w_inside = 2.0       # strong inside penalty
    lambda_smooth = 0.05  # smoothness weight

    iters = 120
    history_stride = 5
    alpha_init = 0.8
    alpha_min = 1e-4

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

    _, grad_edge0, grad_inside0, grad_smooth0 = compute_terms(
        traj_world, sdf, origin_xy, resolution, d_target, margin_inside
    )
    grad_edge_history = [grad_edge0.copy()]
    grad_inside_history = [grad_inside0.copy()]
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
        d_vals, grad_edge, grad_inside, grad_smooth = compute_terms(
            traj_world, sdf, origin_xy, resolution, d_target, margin_inside
        )

        total_grad = (
            w_edge * grad_edge
            + w_inside * grad_inside
            + lambda_smooth * grad_smooth
        )

        # fix endpoints
        total_grad[0] = 0.0
        total_grad[-1] = 0.0

        descent = -total_grad

        # ---- backtracking line search
        alpha = alpha_init

        c0 = total_cost(
            traj_world, sdf, origin_xy,
            resolution, d_target, margin_inside,
            w_edge, w_inside, lambda_smooth
        )

        while True:

            traj_trial = traj_world + alpha * descent

            c_trial = total_cost(
                traj_trial, sdf, origin_xy,
                resolution, d_target, margin_inside,
                w_edge, w_inside, lambda_smooth
            )

            if c_trial < c0:
                break

            alpha *= 0.5

            if alpha < alpha_min:
                # No improving step found; keep current trajectory.
                traj_trial = traj_world.copy()
                break

        traj_world = traj_trial

        if (k + 1) % history_stride == 0:
            traj_history.append(traj_world.copy())
            _, grad_edge_snap, grad_inside_snap, grad_smooth_snap = compute_terms(
                traj_world, sdf, origin_xy, resolution, d_target, margin_inside
            )
            grad_edge_history.append(grad_edge_snap.copy())
            grad_inside_history.append(grad_inside_snap.copy())
            grad_smooth_history.append(grad_smooth_snap.copy())

        if k % 10 == 0:
            c_now = total_cost(
                traj_world, sdf, origin_xy, resolution,
                d_target, margin_inside, w_edge, w_inside, lambda_smooth
            )
            print(
                f"Iter {k}, min dist: {d_vals.min():.3f}, "
                f"cost: {c_now:.4f}, alpha: {alpha:.4f}"
            )


    # =========================================================
    # Visualization 1: trajectory evolution
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

    ax.set_title("Trajectory Progress")
    ax.axis("off")
    ax.legend()

    # =========================================================
    # Visualization 2: gradient history by term
    # =========================================================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_edge, ax_inside, ax_smooth, ax_total = axes.ravel()

    sdf_vis = sdf.copy()
    sdf_vis[obstacle_mask] = -1.0

    def draw_term_history(axh, grad_history, color, title):
        axh.imshow(
            sdf_vis,
            cmap="RdYlGn",
            vmin=-1.0,
            vmax=4.0,
            origin="lower",
        )
        for i, traj in enumerate(traj_history):
            alpha = 0.25 + 0.65 * (i / max(1, len(traj_history) - 1))
            axh.plot(
                traj[:, 0] / resolution,
                traj[:, 1] / resolution,
                color="black",
                alpha=alpha,
                linewidth=1.2,
            )
            axh.scatter(
                traj[:, 0] / resolution,
                traj[:, 1] / resolution,
                color="black",
                s=25,
                alpha=alpha,
            )
            g = grad_history[i]
            axh.quiver(
                traj[:, 0] / resolution,
                traj[:, 1] / resolution,
                g[:, 0],
                g[:, 1],
                color=color,
                angles="xy",
                scale_units="xy",
                width=0.0025,
                alpha=alpha,
            )
        axh.set_title(title)
        axh.axis("off")

    total_grad_history = [
        w_edge * ge + w_inside * gi + lambda_smooth * gs
        for ge, gi, gs in zip(
            grad_edge_history, grad_inside_history, grad_smooth_history
        )
    ]

    draw_term_history(ax_edge, grad_edge_history, "green", "Edge Target Gradient")
    draw_term_history(ax_inside, grad_inside_history, "magenta", "Inside Barrier Gradient")
    draw_term_history(ax_smooth, grad_smooth_history, "blue", "Smoothness Gradient")
    draw_term_history(ax_total, total_grad_history, "red", "Total Gradient")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

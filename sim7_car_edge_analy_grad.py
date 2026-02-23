# sim6_car_edge_h_cost_analy_grad.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query


# =========================================================
# Utility
# =========================================================
def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def generate_horizontal_line(cx_px, cy_px, r_px, width_px, offset_ratio=0.3):
    y = cy_px + int(offset_ratio * r_px)
    xs = np.arange(0, width_px)
    ys = np.full_like(xs, y)
    return xs, ys


def extract_and_downsample(xs, ys, obstacle_mask, stride):
    free_mask = ~obstacle_mask[ys, xs]
    free_idx = np.where(free_mask)[0]
    start = free_idx[0]
    end = free_idx[-1]
    xs_clip = xs[start:end + 1]
    ys_clip = ys[start:end + 1]
    xs_ds = xs_clip[::stride]
    ys_ds = ys_clip[::stride]
    if xs_ds[-1] != xs_clip[-1]:
        xs_ds = np.append(xs_ds, xs_clip[-1])
        ys_ds = np.append(ys_ds, ys_clip[-1])
    return xs_ds, ys_ds


def init_headings_from_path(traj_xy):
    d = np.zeros_like(traj_xy)
    d[1:-1] = traj_xy[2:] - traj_xy[:-2]
    d[0] = traj_xy[1] - traj_xy[0]
    d[-1] = traj_xy[-1] - traj_xy[-2]
    return np.arctan2(d[:, 1], d[:, 0])


# =========================================================
# Obstacle + inside gradient
# =========================================================
def obstacle_grad(traj_xy, beta, sdf, origin_xy,
                  resolution, d_target,
                  margin_inside,
                  car_half_length,
                  car_radius):

    n = traj_xy.shape[0]

    grad_edge_xy = np.zeros_like(traj_xy)
    grad_edge_beta = np.zeros(n)

    grad_inside_xy = np.zeros_like(traj_xy)
    grad_inside_beta = np.zeros(n)

    heading = np.stack([np.cos(beta), np.sin(beta)], axis=1)

    centers_front = traj_xy + car_half_length * heading
    centers_rear  = traj_xy - car_half_length * heading

    d_vals = []

    for i in range(n):

        # --- FRONT CIRCLE ---
        d_f, g_f = sdf_query(
            sdf,
            centers_front[i, 0],
            centers_front[i, 1],
            origin_xy=origin_xy,
            resolution=resolution,
        )

        d_eff_f = d_f - car_radius
        r_f = d_eff_f - d_target

        dc_dbeta_f = car_half_length * np.array(
            [-np.sin(beta[i]), np.cos(beta[i])]
        )

        grad_edge_xy[i] += r_f * g_f
        grad_edge_beta[i] += r_f * g_f.dot(dc_dbeta_f)

        if d_eff_f < margin_inside:
            r2 = d_eff_f - margin_inside
            grad_inside_xy[i] += r2 * g_f
            grad_inside_beta[i] += r2 * g_f.dot(dc_dbeta_f)

        # --- REAR CIRCLE ---
        d_r, g_r = sdf_query(
            sdf,
            centers_rear[i, 0],
            centers_rear[i, 1],
            origin_xy=origin_xy,
            resolution=resolution,
        )

        d_eff_r = d_r - car_radius
        r_r = d_eff_r - d_target

        dc_dbeta_r = -car_half_length * np.array(
            [-np.sin(beta[i]), np.cos(beta[i])]
        )

        grad_edge_xy[i] += r_r * g_r
        grad_edge_beta[i] += r_r * g_r.dot(dc_dbeta_r)

        if d_eff_r < margin_inside:
            r2 = d_eff_r - margin_inside
            grad_inside_xy[i] += r2 * g_r
            grad_inside_beta[i] += r2 * g_r.dot(dc_dbeta_r)

        d_vals.append(min(d_eff_f, d_eff_r))

    return (
        np.array(d_vals),
        grad_edge_xy,
        grad_edge_beta,
        grad_inside_xy,
        grad_inside_beta,
    )

# =========================================================
# Smoothness
# =========================================================
def smoothness_grad(traj_xy):
    grad = np.zeros_like(traj_xy)
    grad[1:-1] = (
        2 * traj_xy[1:-1]
        - traj_xy[:-2]
        - traj_xy[2:]
    )
    return grad


# =========================================================
# H-term (nonholonomic)
# =========================================================
def h_grad(traj_xy, beta):

    n = traj_xy.shape[0]
    grad_xy = np.zeros_like(traj_xy)
    grad_beta = np.zeros(n)

    for k in range(n - 1):

        dx = traj_xy[k + 1, 0] - traj_xy[k, 0]
        dy = traj_xy[k + 1, 1] - traj_xy[k, 1]

        cbk = np.cos(beta[k])
        sbk = np.sin(beta[k])
        cbk1 = np.cos(beta[k + 1])
        sbk1 = np.sin(beta[k + 1])

        h = (cbk + cbk1) * dy - (sbk + sbk1) * dx

        dh_dxk = (sbk + sbk1)
        dh_dyk = -(cbk + cbk1)
        dh_dxk1 = -(sbk + sbk1)
        dh_dyk1 = (cbk + cbk1)

        grad_xy[k, 0] += h * dh_dxk
        grad_xy[k, 1] += h * dh_dyk
        grad_xy[k + 1, 0] += h * dh_dxk1
        grad_xy[k + 1, 1] += h * dh_dyk1

        dh_dbk = (-sbk) * dy - (cbk) * dx
        dh_dbk1 = (-sbk1) * dy - (cbk1) * dx

        grad_beta[k] += h * dh_dbk
        grad_beta[k + 1] += h * dh_dbk1

        # grad_beta *= 2.0

    return grad_xy, grad_beta


def h_cost(traj_xy, beta):
    cost = 0.0
    for k in range(traj_xy.shape[0] - 1):
        dx = traj_xy[k + 1, 0] - traj_xy[k, 0]
        dy = traj_xy[k + 1, 1] - traj_xy[k, 1]
        h = (np.cos(beta[k]) + np.cos(beta[k + 1])) * dy - (
            np.sin(beta[k]) + np.sin(beta[k + 1])
        ) * dx
        cost += 0.5 * h * h
    return cost


def smoothness_cost(traj_xy):
    diffs = traj_xy[1:] - traj_xy[:-1]
    return np.sum(np.linalg.norm(diffs, axis=1) ** 2)


# =========================================================
# MAIN
# =========================================================
def main():

    resolution = 0.02
    d_target = 0.06
    margin_inside = 0.0

    weights = {
        "edge": 0.1,
        "inside": 1.0,   # <-- ADD THIS
        "smooth": 1.0,
        "h": 2.0,
    }


    car_half_length = 0.08
    car_radius = 0.10

    iters = 1200
    history_stride = 5
    alpha_init = 0.05
    alpha_min = 1e-4

    obstacle_mask = load_occupancy_from_png(
        # "circle_obstacle.png",
        "l_shape_obstacle.png",
        obstacle_is_dark=True,
        thresh=200,
    )

    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=5.0,
    )

    h_px, w_px = obstacle_mask.shape
    cx = w_px // 2
    cy = h_px // 2
    r_px = int(1.0 / resolution)

    xs, ys = generate_horizontal_line(cx, cy, r_px, w_px)
    xs_traj, ys_traj = extract_and_downsample(xs, ys, obstacle_mask, 4)

    traj_xy = np.stack(
        [xs_traj * resolution, ys_traj * resolution],
        axis=1
    )

    beta = init_headings_from_path(traj_xy)
    origin_xy = (0.0, 0.0)

    traj_initial = traj_xy.copy()
    traj_history = [traj_xy.copy()]
    grad_edge_history = []
    grad_inside_history = []
    grad_smooth_history = []
    grad_h_history = []

    # =========================================================
    # Optimization loop
    # =========================================================
    for k in range(iters):

        d_vals, grad_edge_xy, grad_edge_beta, grad_inside_xy, grad_inside_beta = obstacle_grad(
            traj_xy, beta, sdf, origin_xy,
            resolution, d_target,
            margin_inside,
            car_half_length,
            car_radius
        )

        grad_smooth_xy = smoothness_grad(traj_xy)
        grad_h_xy, grad_h_beta = h_grad(traj_xy, beta)

        grad_xy = (
            weights["edge"] * grad_edge_xy
            + weights["inside"] * grad_inside_xy
            + weights["smooth"] * grad_smooth_xy
            + weights["h"] * grad_h_xy
        )

        grad_beta = (
            weights["edge"] * grad_edge_beta
            + weights["inside"] * grad_inside_beta
            + weights["h"] *100 * grad_h_beta
        )

        grad_xy[0] = 0
        grad_xy[-1] = 0
        grad_beta[0] = 0
        grad_beta[-1] = 0

        traj_xy -= alpha_init * grad_xy
        beta = wrap_to_pi(beta - alpha_init * grad_beta)

        if k % 20 == 0:
            edge_cost = 0.5 * np.sum((d_vals - d_target) ** 2)
            inside_mask = d_vals < margin_inside
            inside_cost = 0.5 * np.sum(
                (d_vals[inside_mask] - margin_inside) ** 2
            )
            smooth_cost = smoothness_cost(traj_xy)
            h_term_cost = h_cost(traj_xy, beta)
            total_cost = (
                weights["edge"] * edge_cost
                + weights["inside"] * inside_cost
                + weights["smooth"] * smooth_cost
                + weights["h"] * h_term_cost
            )
            print(
                f"Iter {k:03d} | "
                f"edge={edge_cost:.6f}, "
                f"inside={inside_cost:.6f}, "
                f"smooth={smooth_cost:.6f}, "
                f"h={h_term_cost:.6f}, "
                f"total={total_cost:.6f}"
            )

            h_tem_list = []
            dx_temp = traj_xy[1:, 0] - traj_xy[:-1, 0]
            dy_temp = traj_xy[1:, 1] - traj_xy[:-1, 1]
            beta_temp = beta
            for k in range(traj_xy.shape[0] - 1):
                dx = traj_xy[k + 1, 0] - traj_xy[k, 0]
                dy = traj_xy[k + 1, 1] - traj_xy[k, 1]
                h = (np.cos(beta[k]) + np.cos(beta[k + 1])) * dy - (
                    np.sin(beta[k]) + np.sin(beta[k + 1])
                ) * dx
                h_tem_list.append(h)
            print(f"  h-term values: {h_tem_list}")
            print(f"  beta values: {beta_temp}")
            print(f"  dx: {dx_temp}")
            print(f"  dy: {dy_temp}")
            print(f"  grad_h_beta: {grad_h_beta}")
            print(f"  grad_h_xy: {100*grad_h_xy}")

        if (k + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())
            grad_edge_history.append(grad_edge_xy.copy())
            grad_inside_history.append(grad_inside_xy.copy())
            grad_smooth_history.append(grad_smooth_xy.copy())
            grad_h_history.append(grad_h_xy.copy())

    # =========================================================
    # Visualization 1 (SDF + trajectory)
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
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Initial"
    )

    for traj in traj_history[1:-1]:
        ax.plot(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            alpha=0.5,
        )

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
        s=45,
        zorder=6,
    )

    heading = np.stack([np.cos(beta), np.sin(beta)], axis=1)
    front = traj_final + car_half_length * heading
    rear = traj_final - car_half_length * heading

    for pf, pr in zip(front, rear):
        ax.plot(
            [pr[0] / resolution, pf[0] / resolution],
            [pr[1] / resolution, pf[1] / resolution],
            color="cyan",
            linewidth=1.2,
        )
        ax.add_patch(
            Circle(
                (pf[0] / resolution, pf[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
            )
        )
        ax.add_patch(
            Circle(
                (pr[0] / resolution, pr[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
            )
        )

    ax.set_title("Trajectory Progress (Edge + H-term)")
    ax.axis("off")
    ax.legend()

    # =========================================================
    # Gradient history plots
    # =========================================================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_edge, ax_smooth, ax_h, ax_total = axes.ravel()

    def draw_term_history(axh, grad_history, color, title):

        sdf_vis = sdf.copy()
        sdf_vis[obstacle_mask] = -1.0

        axh.imshow(
            sdf_vis,
            cmap="RdYlGn",
            vmin=-1.0,
            vmax=4.0,
            origin="lower",
        )

        axh.plot(
            traj_history[0][:, 0] / resolution,
            traj_history[0][:, 1] / resolution,
            linestyle="--",
            color="black",
            linewidth=1.2,
        )

        for i, traj in enumerate(traj_history[1:], start=1):

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

            g = grad_history[i - 1]

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

    draw_term_history(ax_edge, grad_edge_history, "green", "Edge Term Gradient")
    draw_term_history(ax_smooth, grad_smooth_history, "blue", "Smoothness Term Gradient")
    draw_term_history(ax_h, grad_h_history, "magenta", "H-Term Gradient")

    total_hist = [
        weights["edge"] * ge
        + weights["inside"] * gi
        + weights["smooth"] * gs
        + weights["h"] * gh
        for ge, gi, gs, gh in zip(
            grad_edge_history,
            grad_inside_history,
            grad_smooth_history,
            grad_h_history
        )
    ]

    draw_term_history(ax_total, total_hist, "red", "Total Gradient")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

    xs_clip = xs[start:end + 1]
    ys_clip = ys[start:end + 1]

    xs_ds = xs_clip[::stride]
    ys_ds = ys_clip[::stride]

    if xs_ds[-1] != xs_clip[-1]:
        xs_ds = np.append(xs_ds, xs_clip[-1])
        ys_ds = np.append(ys_ds, ys_clip[-1])

    return xs_ds, ys_ds


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def init_headings_from_path(traj_xy):
    n = traj_xy.shape[0]
    beta = np.zeros(n, dtype=float)
    if n < 2:
        return beta

    d = np.zeros_like(traj_xy)
    d[1:-1] = traj_xy[2:] - traj_xy[:-2]
    d[0] = traj_xy[1] - traj_xy[0]
    d[-1] = traj_xy[-1] - traj_xy[-2]

    beta = np.arctan2(d[:, 1], d[:, 0])
    return beta


def front_rear_centers(traj_xy, beta, car_half_length):
    heading = np.stack([np.cos(beta), np.sin(beta)], axis=1)
    centers_front = traj_xy + car_half_length * heading
    centers_rear = traj_xy - car_half_length * heading
    return centers_front, centers_rear


def query_sdf_points(points_world, sdf, origin_xy, resolution):
    d_vals = []
    grads = []
    for x, y in points_world:
        d, g = sdf_query(
            sdf,
            x,
            y,
            origin_xy=origin_xy,
            resolution=resolution,
        )
        d_vals.append(d)
        grads.append(g)
    return np.array(d_vals), np.array(grads)


def compute_curvature(traj_xy, beta, eps_ds=1e-6):
    dxy = traj_xy[1:] - traj_xy[:-1]
    ds = np.linalg.norm(dxy, axis=1)
    ds_safe = np.maximum(ds, eps_ds)
    d_beta = wrap_to_pi(beta[1:] - beta[:-1])

    kappa = 2.0 * np.sin(0.5 * d_beta) / ds_safe
    return kappa, dxy, ds_safe, d_beta


def compute_h_residual(traj_xy, beta):
    dx = traj_xy[1:, 0] - traj_xy[:-1, 0]
    dy = traj_xy[1:, 1] - traj_xy[:-1, 1]

    qsum_x = np.cos(beta[:-1]) + np.cos(beta[1:])
    qsum_y = np.sin(beta[:-1]) + np.sin(beta[1:])

    # Strict Eq. (4): h_k = (q_k + q_{k+1}) x d_k
    h = qsum_x * dy - qsum_y * dx
    return h


def total_cost_components(
    traj_xy,
    beta,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
    car_half_length,
    car_radius,
    kappa_max,
):
    centers_front, centers_rear = front_rear_centers(traj_xy, beta, car_half_length)

    d_front, _ = query_sdf_points(centers_front, sdf, origin_xy, resolution)
    d_rear, _ = query_sdf_points(centers_rear, sdf, origin_xy, resolution)

    d_eff_front = d_front - car_radius
    d_eff_rear = d_rear - car_radius

    edge_cost = 0.5 * np.sum((d_eff_front - d_target) ** 2)
    edge_cost += 0.5 * np.sum((d_eff_rear - d_target) ** 2)

    inside_mask_front = d_eff_front < margin_inside
    inside_mask_rear = d_eff_rear < margin_inside
    inside_cost = 0.5 * np.sum((d_eff_front[inside_mask_front] - margin_inside) ** 2)
    inside_cost += 0.5 * np.sum((d_eff_rear[inside_mask_rear] - margin_inside) ** 2)

    smooth_pos_cost = np.sum(np.linalg.norm(traj_xy[1:] - traj_xy[:-1], axis=1) ** 2)

    h = compute_h_residual(traj_xy, beta)
    # print("h residuals:", h)
    h_cost = 0.5 * np.sum(h ** 2)

    kappa, _, _, _ = compute_curvature(traj_xy, beta)
    kappa_violation = np.maximum(0.0, np.abs(kappa) - kappa_max)
    kappa_bound_cost = 0.5 * np.sum(kappa_violation ** 2)

    if kappa.size >= 2:
        kappa_smooth_cost = 0.5 * np.sum((kappa[1:] - kappa[:-1]) ** 2)
    else:
        kappa_smooth_cost = 0.0

    d_min_eff = np.minimum(d_eff_front, d_eff_rear)

    return {
        "edge": edge_cost,
        "inside": inside_cost,
        "smooth_pos": smooth_pos_cost,
        "h": h_cost,
        "kappa_bound": kappa_bound_cost,
        "kappa_smooth": kappa_smooth_cost,
        "d_min_eff": d_min_eff,
        "centers_front": centers_front,
        "centers_rear": centers_rear,
        "kappa": kappa,
        "h_residual": h,
    }


def weighted_total_cost(comps, weights):
    return (
        weights["edge"] * comps["edge"]
        + weights["inside"] * comps["inside"]
        + weights["smooth_pos"] * comps["smooth_pos"]
        + weights["h"] * comps["h"]
        + weights["kappa_bound"] * comps["kappa_bound"]
        + weights["kappa_smooth"] * comps["kappa_smooth"]
    )


def finite_difference_grads(
    traj_xy,
    beta,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
    car_half_length,
    car_radius,
    kappa_max,
    eps_xy,
    eps_beta,
):
    base = total_cost_components(
        traj_xy,
        beta,
        sdf,
        origin_xy,
        resolution,
        d_target,
        margin_inside,
        car_half_length,
        car_radius,
        kappa_max,
    )

    n = traj_xy.shape[0]
    terms = ["edge", "inside", "smooth_pos", "h", "kappa_bound", "kappa_smooth"]

    grad_xy_terms = {t: np.zeros_like(traj_xy) for t in terms}
    grad_beta_terms = {t: np.zeros(n, dtype=float) for t in terms}

    for i in range(n):
        for j in range(2):
            traj_p = traj_xy.copy()
            traj_p[i, j] += eps_xy

            comps_p = total_cost_components(
                traj_p,
                beta,
                sdf,
                origin_xy,
                resolution,
                d_target,
                margin_inside,
                car_half_length,
                car_radius,
                kappa_max,
            )

            for t in terms:
                grad_xy_terms[t][i, j] = (comps_p[t] - base[t]) / eps_xy

        beta_p = beta.copy()
        beta_p[i] += eps_beta

        comps_p = total_cost_components(
            traj_xy,
            beta_p,
            sdf,
            origin_xy,
            resolution,
            d_target,
            margin_inside,
            car_half_length,
            car_radius,
            kappa_max,
        )

        for t in terms:
            grad_beta_terms[t][i] = (comps_p[t] - base[t]) / eps_beta

    return base, grad_xy_terms, grad_beta_terms


def main():
    resolution = 0.02

    # ---- Tuning block
    d_target = 0.06
    margin_inside = 0.0

    weights = {
        "edge": 1.0,
        "inside": 1.0,
        "smooth_pos": 0.5,
        "h": 10.0,
        "kappa_bound": 0.5,
        "kappa_smooth": 0.00,
    }

    car_half_length = 0.08
    car_radius = 0.10

    rho_min = 0.35
    kappa_max = 1.0 / rho_min

    iters = 21
    history_stride = 5
    alpha_init = 0.3
    alpha_min = 1e-4

    eps_xy = 1e-4
    eps_beta = 1e-4

    obstacle_mask = load_occupancy_from_png(
        "circle_obstacle.png",
        obstacle_is_dark=True,
        thresh=200,
    )

    h_px, w_px = obstacle_mask.shape

    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=5.0,
    )

    cx = w_px // 2
    cy = h_px // 2
    r_px = int(1.0 / resolution)

    xs, ys = generate_horizontal_line(cx, cy, r_px, w_px)
    xs_traj, ys_traj = extract_and_downsample(xs, ys, obstacle_mask, stride=10)

    traj_xy = np.stack([xs_traj * resolution, ys_traj * resolution], axis=1)
    beta = init_headings_from_path(traj_xy)

    traj_initial = traj_xy.copy()
    beta_initial = beta.copy()

    traj_history = [traj_xy.copy()]
    beta_history = [beta.copy()]

    origin_xy = (0.0, 0.0)

    base0, grad_xy_terms0, grad_beta_terms0 = finite_difference_grads(
        traj_xy,
        beta,
        sdf,
        origin_xy,
        resolution,
        d_target,
        margin_inside,
        car_half_length,
        car_radius,
        kappa_max,
        eps_xy,
        eps_beta,
    )

    grad_edge_history = [grad_xy_terms0["edge"].copy()]
    grad_inside_history = [grad_xy_terms0["inside"].copy()]
    grad_beta_edge_history = [grad_beta_terms0["edge"].copy()]
    grad_beta_inside_history = [grad_beta_terms0["inside"].copy()]
    grad_beta_h_history = [grad_beta_terms0["h"].copy()]
    grad_beta_kappa_history = [
        (
            weights["kappa_bound"] * grad_beta_terms0["kappa_bound"]
            + weights["kappa_smooth"] * grad_beta_terms0["kappa_smooth"]
        ).copy()
    ]

    grad_reg0 = (
        weights["smooth_pos"] * grad_xy_terms0["smooth_pos"]
        + weights["h"] * grad_xy_terms0["h"]
        + weights["kappa_bound"] * grad_xy_terms0["kappa_bound"]
        + weights["kappa_smooth"] * grad_xy_terms0["kappa_smooth"]
    )
    grad_reg_history = [grad_reg0.copy()]

    grad_total0 = (
        weights["edge"] * grad_xy_terms0["edge"]
        + weights["inside"] * grad_xy_terms0["inside"]
        + grad_reg0
    )
    grad_total_history = [grad_total0.copy()]
    grad_beta_total0 = (
        weights["edge"] * grad_beta_terms0["edge"]
        + weights["inside"] * grad_beta_terms0["inside"]
        + weights["h"] * grad_beta_terms0["h"]
        + weights["kappa_bound"] * grad_beta_terms0["kappa_bound"]
        + weights["kappa_smooth"] * grad_beta_terms0["kappa_smooth"]
    )
    grad_beta_total_history = [grad_beta_total0.copy()]

    # =========================================================
    # Optimization loop (first-order with line search)
    # =========================================================
    for k in range(iters):
        base, grad_xy_terms, grad_beta_terms = finite_difference_grads(
            traj_xy,
            beta,
            sdf,
            origin_xy,
            resolution,
            d_target,
            margin_inside,
            car_half_length,
            car_radius,
            kappa_max,
            eps_xy,
            eps_beta,
        )

        grad_xy_total = (
            weights["edge"] * grad_xy_terms["edge"]
            + weights["inside"] * grad_xy_terms["inside"]
            + weights["smooth_pos"] * grad_xy_terms["smooth_pos"]
            + weights["h"] * grad_xy_terms["h"]
            + weights["kappa_bound"] * grad_xy_terms["kappa_bound"]
            + weights["kappa_smooth"] * grad_xy_terms["kappa_smooth"]
        )

        grad_beta_total = (
            weights["edge"] * grad_beta_terms["edge"]
            + weights["inside"] * grad_beta_terms["inside"]
            + weights["smooth_pos"] * grad_beta_terms["smooth_pos"]
            + weights["h"] * grad_beta_terms["h"]
            + weights["kappa_bound"] * grad_beta_terms["kappa_bound"]
            + weights["kappa_smooth"] * grad_beta_terms["kappa_smooth"]
        )

        grad_xy_total[0] = 0.0
        grad_xy_total[-1] = 0.0
        grad_beta_total[0] = 0.0
        grad_beta_total[-1] = 0.0

        descent_xy = -grad_xy_total
        descent_beta = -grad_beta_total

        alpha = alpha_init
        c0 = weighted_total_cost(base, weights)

        while True:
            traj_trial = traj_xy + alpha * descent_xy
            beta_trial = wrap_to_pi(beta + alpha * descent_beta)

            comps_trial = total_cost_components(
                traj_trial,
                beta_trial,
                sdf,
                origin_xy,
                resolution,
                d_target,
                margin_inside,
                car_half_length,
                car_radius,
                kappa_max,
            )
            c_trial = weighted_total_cost(comps_trial, weights)

            if c_trial < c0:
                break

            alpha *= 0.5
            if alpha < alpha_min:
                traj_trial = traj_xy.copy()
                beta_trial = beta.copy()
                comps_trial = base
                break

        traj_xy = traj_trial
        beta = beta_trial

        if (k + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())
            beta_history.append(beta.copy())

            _, grad_xy_snap, grad_beta_snap = finite_difference_grads(
                traj_xy,
                beta,
                sdf,
                origin_xy,
                resolution,
                d_target,
                margin_inside,
                car_half_length,
                car_radius,
                kappa_max,
                eps_xy,
                eps_beta,
            )

            grad_edge_history.append(grad_xy_snap["edge"].copy())
            grad_inside_history.append(grad_xy_snap["inside"].copy())
            grad_beta_edge_history.append(grad_beta_snap["edge"].copy())
            grad_beta_inside_history.append(grad_beta_snap["inside"].copy())
            grad_beta_h_history.append(grad_beta_snap["h"].copy())

            grad_beta_kappa_snap = (
                weights["kappa_bound"] * grad_beta_snap["kappa_bound"]
                + weights["kappa_smooth"] * grad_beta_snap["kappa_smooth"]
            )
            grad_beta_kappa_history.append(grad_beta_kappa_snap.copy())

            grad_reg_snap = (
                weights["smooth_pos"] * grad_xy_snap["smooth_pos"]
                + weights["h"] * grad_xy_snap["h"]
                + weights["kappa_bound"] * grad_xy_snap["kappa_bound"]
                + weights["kappa_smooth"] * grad_xy_snap["kappa_smooth"]
            )
            grad_reg_history.append(grad_reg_snap.copy())

            grad_total_snap = (
                weights["edge"] * grad_xy_snap["edge"]
                + weights["inside"] * grad_xy_snap["inside"]
                + grad_reg_snap
            )
            grad_total_history.append(grad_total_snap.copy())
            grad_beta_total_snap = (
                weights["edge"] * grad_beta_snap["edge"]
                + weights["inside"] * grad_beta_snap["inside"]
                + weights["h"] * grad_beta_snap["h"]
                + weights["kappa_bound"] * grad_beta_snap["kappa_bound"]
                + weights["kappa_smooth"] * grad_beta_snap["kappa_smooth"]
            )
            grad_beta_total_history.append(grad_beta_total_snap.copy())

        if k % 10 == 0:
            kappa_now = comps_trial["kappa"]
            h_now = comps_trial["h_residual"]
            print(
                f"Iter {k}, min d_eff: {comps_trial['d_min_eff'].min():.3f}, "
                f"max |kappa|: {np.max(np.abs(kappa_now)) if kappa_now.size else 0.0:.3f}, "
                f"max |h|: {np.max(np.abs(h_now)) if h_now.size else 0.0:.3f}, "
                f"|grad_beta_h|: {np.linalg.norm(grad_beta_terms['h']):.3e}, "
                f"cost: {c_trial:.4f}, alpha: {alpha:.4f}"
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

    ax.plot(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Initial",
    )

    ax.scatter(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        color="black",
        s=70,
        zorder=5,
    )

    for i, traj in enumerate(traj_history[1:-1], start=1):
        alpha_hist = 0.3 + 0.5 * (i / max(1, len(traj_history)))
        ax.plot(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            alpha=alpha_hist,
            linewidth=1.5,
        )
        ax.scatter(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            s=40,
            alpha=alpha_hist,
        )

    traj_final = traj_history[-1]
    beta_final = beta_history[-1]

    ax.plot(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        linewidth=3,
        label="Final",
    )

    ax.scatter(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        s=80,
        zorder=6,
    )

    # Visualize heading angle beta at each final waypoint.
    heading_len_px = 0.8 * car_half_length / resolution
    hx = np.cos(beta_final) * heading_len_px
    hy = np.sin(beta_final) * heading_len_px
    ax.quiver(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        hx,
        hy,
        color="orange",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        alpha=0.9,
        zorder=7,
    )

    front_final, rear_final = front_rear_centers(traj_final, beta_final, car_half_length)
    for pf, pr in zip(front_final, rear_final):
        ax.plot(
            [pr[0] / resolution, pf[0] / resolution],
            [pr[1] / resolution, pf[1] / resolution],
            color="cyan",
            linewidth=1.2,
            alpha=0.7,
        )
        ax.add_patch(
            Circle(
                (pf[0] / resolution, pf[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
                alpha=0.7,
            )
        )
        ax.add_patch(
            Circle(
                (pr[0] / resolution, pr[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
                alpha=0.7,
            )
        )

    ax.set_title("Trajectory Progress (Curvature/Heading State)")
    ax.axis("off")
    ax.legend()

    # =========================================================
    # Visualization 2: gradient history by term
    # =========================================================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_edge, ax_inside, ax_reg, ax_total = axes.ravel()

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
            alpha_hist = 0.25 + 0.65 * (i / max(1, len(traj_history) - 1))
            axh.plot(
                traj[:, 0] / resolution,
                traj[:, 1] / resolution,
                color="black",
                alpha=alpha_hist,
                linewidth=1.2,
            )
            axh.scatter(
                traj[:, 0] / resolution,
                traj[:, 1] / resolution,
                color="black",
                s=25,
                alpha=alpha_hist,
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
                alpha=alpha_hist,
            )
        axh.set_title(title)
        axh.axis("off")

    draw_term_history(ax_edge, grad_edge_history, "green", "Edge Target Gradient")
    draw_term_history(ax_inside, grad_inside_history, "magenta", "Inside Barrier Gradient")
    draw_term_history(ax_reg, grad_reg_history, "blue", "Regularization Gradient")
    draw_term_history(ax_total, grad_total_history, "red", "Total Gradient")

    fig2.tight_layout()

    # =========================================================
    # Visualization 3: beta-gradient history by term
    # =========================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    ax_b_edge, ax_b_h, ax_b_kappa, ax_b_total = axes3.ravel()

    def draw_beta_history(axh, beta_grad_history, title, color):
        x = np.arange(beta_grad_history[0].shape[0])
        for i, gb in enumerate(beta_grad_history):
            alpha_hist = 0.25 + 0.65 * (i / max(1, len(beta_grad_history) - 1))
            axh.plot(x, gb, color=color, alpha=alpha_hist, linewidth=1.2)
        axh.set_title(title)
        axh.set_xlabel("Waypoint index")
        axh.set_ylabel("dCost / d beta")
        axh.grid(True, alpha=0.3)

    draw_beta_history(ax_b_edge, grad_beta_edge_history, "Beta Grad: Edge", "green")
    draw_beta_history(
        ax_b_h,
        [weights["h"] * g for g in grad_beta_h_history],
        "Beta Grad: h-term (weighted)",
        "magenta",
    )
    draw_beta_history(ax_b_kappa, grad_beta_kappa_history, "Beta Grad: Curvature (weighted)", "blue")
    draw_beta_history(ax_b_total, grad_beta_total_history, "Beta Grad: Total", "red")

    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

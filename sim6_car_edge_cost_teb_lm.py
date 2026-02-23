import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import least_squares

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d

# ---- import core functions from your previous file ----
from sim5_car_edge_cost_teb import (
    total_cost_components,
    init_headings_from_path,
    wrap_to_pi,
    front_rear_centers,
)

# =========================================================
# Packing utilities
# =========================================================

def pack_state(traj_xy, beta):
    return np.concatenate([traj_xy.flatten(), beta])


def unpack_state(x, n):
    traj_xy = x[: 2 * n].reshape(n, 2)
    beta = x[2 * n:]
    return traj_xy, beta


# =========================================================
# Residual vector (for Gauss–Newton / LM)
# =========================================================

def residual_vector(
    x,
    n,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
    car_half_length,
    car_radius,
    kappa_max,
    weights,
):
    traj_xy, beta = unpack_state(x, n)

    comps = total_cost_components(
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

    residuals = []

    # Edge residual
    residuals.append(
        np.sqrt(weights["edge"]) *
        (comps["d_min_eff"] - d_target)
    )

    # Inside residual
    inside_violation = np.minimum(
        0.0, comps["d_min_eff"] - margin_inside
    )
    residuals.append(
        np.sqrt(weights["inside"]) *
        inside_violation
    )

    # h constraint
    residuals.append(
        np.sqrt(weights["h"]) *
        comps["h_residual"]
    )

    # Position smoothness
    dxy = traj_xy[1:] - traj_xy[:-1]
    residuals.append(
        np.sqrt(weights["smooth_pos"]) *
        dxy.flatten()
    )

    # Curvature bound
    kappa_violation = np.maximum(
        0.0, np.abs(comps["kappa"]) - kappa_max
    )
    residuals.append(
        np.sqrt(weights["kappa_bound"]) *
        kappa_violation
    )

    # Curvature smoothness
    if comps["kappa"].size >= 2:
        dk = comps["kappa"][1:] - comps["kappa"][:-1]
        residuals.append(
            np.sqrt(weights["kappa_smooth"]) *
            dk
        )

    return np.concatenate(residuals)


# =========================================================
# Main
# =========================================================

def main():

    resolution = 0.02

    d_target = 0.06
    margin_inside = 0.0

    weights = {
        "edge": 1.0,
        "inside": 1.0,
        "smooth_pos": 1.0,
        "h": 0.0,
        "kappa_bound": 0.5,
        "kappa_smooth": 0.0,
    }

    car_half_length = 0.08
    car_radius = 0.10

    rho_min = 0.35
    kappa_max = 1.0 / rho_min

    obstacle_mask = load_occupancy_from_png(
        "circle_obstacle.png",
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

    xs = np.arange(0, w_px)
    ys = np.full_like(xs, cy)

    free_mask = ~obstacle_mask[ys, xs]
    free_idx = np.where(free_mask)[0]
    xs = xs[free_idx[0]:free_idx[-1]]
    ys = ys[free_idx[0]:free_idx[-1]]

    stride = 10
    xs = xs[::stride]
    ys = ys[::stride]

    traj_xy = np.stack(
        [xs * resolution, ys * resolution],
        axis=1,
    )

    beta = init_headings_from_path(traj_xy)

    traj_initial = traj_xy.copy()
    beta_initial = beta.copy()

    n = traj_xy.shape[0]
    x0 = pack_state(traj_xy, beta)

    print("Starting LM / Gauss–Newton optimization...")

    result = least_squares(
        residual_vector,
        x0,
        method="trf",     # trust-region Gauss–Newton
        verbose=2,
        max_nfev=50,
        args=(
            n,
            sdf,
            (0.0, 0.0),
            resolution,
            d_target,
            margin_inside,
            car_half_length,
            car_radius,
            kappa_max,
            weights,
        ),
    )

    traj_xy, beta = unpack_state(result.x, n)
    beta = wrap_to_pi(beta)

    print("Final cost:", result.cost)

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

    ax.plot(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Initial",
    )

    ax.plot(
        traj_xy[:, 0] / resolution,
        traj_xy[:, 1] / resolution,
        color="blue",
        linewidth=3,
        label="Final",
    )

    ax.scatter(
        traj_xy[:, 0] / resolution,
        traj_xy[:, 1] / resolution,
        color="blue",
        s=60,
    )

    # Heading arrows
    heading_len_px = 0.8 * car_half_length / resolution
    hx = np.cos(beta) * heading_len_px
    hy = np.sin(beta) * heading_len_px

    ax.quiver(
        traj_xy[:, 0] / resolution,
        traj_xy[:, 1] / resolution,
        hx,
        hy,
        color="orange",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )

    # Car footprint
    front_final, rear_final = front_rear_centers(
        traj_xy, beta, car_half_length
    )

    for pf, pr in zip(front_final, rear_final):
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
            )
        )
        ax.add_patch(
            Circle(
                (pr[0] / resolution, pr[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
            )
        )

    ax.set_title("Trajectory Optimization (LM / Gauss–Newton)")
    ax.axis("off")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()

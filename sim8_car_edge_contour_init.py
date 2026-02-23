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


def init_headings_from_path(traj_xy):
    d = np.zeros_like(traj_xy)
    d[1:-1] = traj_xy[2:] - traj_xy[:-2]
    d[0] = traj_xy[1] - traj_xy[0]
    d[-1] = traj_xy[-1] - traj_xy[-2]
    return np.arctan2(d[:, 1], d[:, 0])


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required. Install with: pip install opencv-python"
        ) from exc
    return cv2


def _require_pyclipper():
    try:
        import pyclipper  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyclipper is required. Install with: pip install pyclipper"
        ) from exc
    return pyclipper


def extract_largest_contour_px(obstacle_mask_bool, min_area_px2=20.0):
    cv2 = _require_cv2()
    mask_u8 = obstacle_mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour found in obstacle mask.")

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    if areas[idx] < min_area_px2:
        raise RuntimeError(f"Largest contour area too small: {areas[idx]:.3f}")
    return contours[idx]


def dilate_contour_px(contour_px, radius_px, join_type="round"):
    if radius_px <= 0.0:
        return contour_px.copy()

    pyclipper = _require_pyclipper()
    pts = contour_px.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 contour points for dilation.")

    join_map = {
        "round": pyclipper.JT_ROUND,
        "miter": pyclipper.JT_MITER,
        "square": pyclipper.JT_SQUARE,
    }
    if join_type not in join_map:
        raise ValueError(f"Unsupported join_type: {join_type}")

    scale = 1000.0
    path = [(int(round(p[0] * scale)), int(round(p[1] * scale))) for p in pts]

    co = pyclipper.PyclipperOffset()
    co.ArcTolerance = 0.25 * scale
    co.AddPath(path, join_map[join_type], pyclipper.ET_CLOSEDPOLYGON)
    solution = co.Execute(float(radius_px) * scale)
    if not solution:
        raise RuntimeError("pyclipper offset produced no output contour.")

    best = max(solution, key=lambda s: abs(pyclipper.Area(s)))
    best_np = np.asarray(best, dtype=float) / scale
    return best_np.reshape(-1, 1, 2)


def resample_closed_contour_uniform_px(contour_px, step_px):
    pts = contour_px.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        raise ValueError("Contour needs >=3 points to sample.")
    if step_px <= 0:
        raise ValueError("step_px must be > 0")

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    pts0 = pts[:-1][keep]
    pts1 = pts[1:][keep]
    seg_len = seg_len[keep]

    if seg_len.size == 0:
        raise RuntimeError("Degenerate contour with zero-length segments.")

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = cum[-1]

    n_samples = max(8, int(np.floor(perimeter / step_px)))
    s_vals = np.linspace(0.0, perimeter, n_samples, endpoint=False)

    out = np.zeros((n_samples, 2), dtype=float)
    j = 0
    for i, s in enumerate(s_vals):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        ds = s - cum[j]
        t = ds / max(seg_len[j], 1e-9)
        out[i] = (1.0 - t) * pts0[j] + t * pts1[j]

    return out


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
    centers_rear = traj_xy - car_half_length * heading

    d_vals = []

    for i in range(n):

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
    grad[1:-1] = 2 * traj_xy[1:-1] - traj_xy[:-2] - traj_xy[2:]
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

        dh_dxk = sbk + sbk1
        dh_dyk = -(cbk + cbk1)
        dh_dxk1 = -(sbk + sbk1)
        dh_dyk1 = cbk + cbk1

        grad_xy[k, 0] += h * dh_dxk
        grad_xy[k, 1] += h * dh_dyk
        grad_xy[k + 1, 0] += h * dh_dxk1
        grad_xy[k + 1, 1] += h * dh_dyk1

        dh_dbk = (-sbk) * dy - cbk * dx
        dh_dbk1 = (-sbk1) * dy - cbk1 * dx

        grad_beta[k] += h * dh_dbk
        grad_beta[k + 1] += h * dh_dbk1

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

    resolution = 0.05
    # d_target = 0.06
    d_target = 0.06
    margin_inside = 0.0

    # weights = {
    #     "edge": 1.0,
    #     "inside": 2.0,
    #     "smooth": 1.0,
    #     "h": 2.0,
    # }

    weights = {
        "edge": 0.0,
        "inside": 2.0,
        "smooth": 0.0,
        "h": 2.0,
    }

    car_half_length = 0.2
    car_radius = 0.2

    # Sampling resolution along contour in meters.
    sample_step_m = 0.05

    # Dilation radius equals car circle radius (in pixels).
    dilate_radius_px = car_radius / resolution

    iters = 201
    history_stride = 5
    alpha_init = 0.05

    obstacle_mask = load_occupancy_from_png(
        # "circle_obstacle.png",
        # "l_shape_obstacle.png",
        "l_shape_obstacle_45deg.png",
  
        obstacle_is_dark=True,
        thresh=200,
    )

    sdf = signed_esdf_2d(
        obstacle_mask,
        resolution=resolution,
        max_dist=5.0,
    )

    contour_px = extract_largest_contour_px(obstacle_mask)
    dilated_px = dilate_contour_px(contour_px, dilate_radius_px, join_type="round")

    sample_step_px = sample_step_m / resolution
    traj_px = resample_closed_contour_uniform_px(dilated_px, sample_step_px)

    # Close the loop by repeating first point as the last point.
    traj_px = np.vstack([traj_px, traj_px[0]])

    traj_xy = traj_px * resolution
    beta = init_headings_from_path(traj_xy)
    origin_xy = (0.0, 0.0)

    print(f"Contour points: {len(contour_px)}")
    print(f"Dilated contour points: {len(dilated_px)}")
    print(f"Dilate radius: {dilate_radius_px:.2f} px ({car_radius:.3f} m)")
    print(f"Sampling step: {sample_step_px:.2f} px ({sample_step_m:.3f} m)")
    print(f"Trajectory samples (closed): {len(traj_xy)}")

    traj_initial = traj_xy.copy()
    traj_history = [traj_xy.copy()]

    for k in range(iters):

        d_vals, grad_edge_xy, grad_edge_beta, grad_inside_xy, grad_inside_beta = obstacle_grad(
            traj_xy,
            beta,
            sdf,
            origin_xy,
            resolution,
            d_target,
            margin_inside,
            car_half_length,
            car_radius,
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
            + weights["h"] * 100 * grad_h_beta
        )

        # Keep loop seam fixed.
        grad_xy[0] = 0
        grad_xy[-1] = 0
        grad_beta[0] = 0
        grad_beta[-1] = 0

        traj_xy -= alpha_init * grad_xy
        beta = wrap_to_pi(beta - alpha_init * grad_beta)

        if k % 50 == 0:
            edge_cost = 0.5 * np.sum((d_vals - d_target) ** 2)
            inside_mask = d_vals < margin_inside
            inside_cost = 0.5 * np.sum((d_vals[inside_mask] - margin_inside) ** 2)
            smooth_cost = smoothness_cost(traj_xy)
            h_term_cost = h_cost(traj_xy, beta)
            total_cost = (
                weights["edge"] * edge_cost
                + weights["inside"] * inside_cost
                + weights["smooth"] * smooth_cost
                + weights["h"] * h_term_cost
            )
            print(
                f"Iter {k:04d} | edge={edge_cost:.6f}, inside={inside_cost:.6f}, "
                f"smooth={smooth_cost:.6f}, h={h_term_cost:.6f}, total={total_cost:.6f}"
            )

        if (k + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())

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

    cpts = contour_px.reshape(-1, 2)
    dpts = dilated_px.reshape(-1, 2)
    ax.plot(cpts[:, 0], cpts[:, 1], "w--", linewidth=1.5, label="Obstacle contour")
    ax.plot(dpts[:, 0], dpts[:, 1], "y-", linewidth=1.5, label="Dilated contour (init source)")

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
            alpha=0.35,
        )

    traj_final = traj_history[-1]

    ax.plot(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        linewidth=2.5,
        label="Final",
    )
    ax.scatter(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="red",
        s=18,
        alpha=0.95,
        zorder=7,
        label="Final waypoints",
    )

    heading = np.stack([np.cos(beta), np.sin(beta)], axis=1)
    front = traj_final + car_half_length * heading
    rear = traj_final - car_half_length * heading

    for pf, pr in zip(front, rear):
        ax.plot(
            [pr[0] / resolution, pf[0] / resolution],
            [pr[1] / resolution, pf[1] / resolution],
            color="cyan",
            linewidth=0.8,
        )
        ax.add_patch(
            Circle(
                (pf[0] / resolution, pf[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=0.8,
            )
        )
        ax.add_patch(
            Circle(
                (pr[0] / resolution, pr[1] / resolution),
                radius=car_radius / resolution,
                fill=False,
                edgecolor="cyan",
                linewidth=0.8,
            )
        )

    ax.set_title("sim8: Contour-Dilated Initialization")
    ax.axis("off")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("\nFinal waypoints:")
    print(f"Waypoint count: {len(traj_final)}")
    print("idx, x[m], y[m], beta[rad]")
    for i in range(len(traj_final)):
        print(
            f"{i:03d}, "
            f"{traj_final[i, 0]:.6f}, "
            f"{traj_final[i, 1]:.6f}, "
            f"{beta[i]:.6f}"
        )


if __name__ == "__main__":
    main()

# sim8_mower_front_edge_follow.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query

from mower_model import MowerFrontModel


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
# Smoothness
# =========================================================
def smoothness_grad(traj_xy):
    grad = np.zeros_like(traj_xy)
    grad[1:-1] = 2 * traj_xy[1:-1] - traj_xy[:-2] - traj_xy[2:]
    return grad


# =========================================================
# Nonholonomic term
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


# =========================================================
# 1️⃣ Safety term
# =========================================================
def obstacle_safety_grad(
    traj_xy,
    beta,
    robot: MowerFrontModel,
    sdf,
    *,
    origin_xy,
    resolution,
    d_safe: float,
):
    n = traj_xy.shape[0]
    grad_xy = np.zeros_like(traj_xy)
    grad_beta = np.zeros(n)

    for i in range(n):
        state = (float(traj_xy[i, 0]), float(traj_xy[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)

        for pr in prims:
            c = pr["center_w"]
            r = pr["radius"]
            J_theta = pr["J_theta"]

            d, g = sdf_query(
                sdf,
                float(c[0]),
                float(c[1]),
                origin_xy=origin_xy,
                resolution=resolution,
            )

            d_eff = d - r

            if d_eff < d_safe:
                rs = (d_eff - d_safe)
                grad_xy[i] += rs * g
                grad_beta[i] += rs * g.dot(J_theta)

    return grad_xy, grad_beta


# =========================================================
# 2️⃣ Edge term
# =========================================================
def edge_band_grad(
    traj_xy,
    beta,
    robot: MowerFrontModel,
    sdf,
    *,
    origin_xy,
    resolution,
    d_edge: float,
):
    n = traj_xy.shape[0]
    grad_xy = np.zeros_like(traj_xy)
    grad_beta = np.zeros(n)

    for i in range(n):
        state = (float(traj_xy[i, 0]), float(traj_xy[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)

        for pr in prims:
            if pr["tag"] != "working":
                continue

            c = pr["center_w"]
            r = pr["radius"]
            J_theta = pr["J_theta"]

            d, g = sdf_query(
                sdf,
                float(c[0]),
                float(c[1]),
                origin_xy=origin_xy,
                resolution=resolution,
            )

            d_eff = d - r
            re = (d_eff - d_edge)

            grad_xy[i] += re * g
            grad_beta[i] += re * g.dot(J_theta)

    return grad_xy, grad_beta


# =========================================================
# 3️⃣ Tangent term
# =========================================================
def tangent_motion_grad(
    traj_xy,
    beta,
    robot: MowerFrontModel,
    sdf,
    *,
    origin_xy,
    resolution,
    w_tangent: float,
):
    n = traj_xy.shape[0]
    grad_xy = np.zeros_like(traj_xy)
    grad_beta = np.zeros(n)

    if w_tangent == 0.0:
        return grad_xy, grad_beta

    for k in range(n - 1):
        state = (float(traj_xy[k, 0]), float(traj_xy[k, 1]), float(beta[k]))
        prims = robot.world_centers_and_jacobians(state)

        deck = None
        for pr in prims:
            if pr["tag"] == "working":
                deck = pr
                break

        if deck is None:
            continue

        c = deck["center_w"]
        d, g = sdf_query(
            sdf,
            float(c[0]),
            float(c[1]),
            origin_xy=origin_xy,
            resolution=resolution,
        )

        n_hat = g / (np.linalg.norm(g) + 1e-12)
        v = traj_xy[k + 1] - traj_xy[k]
        vn = float(v.dot(n_hat))

        grad_xy[k]     += -w_tangent * vn * n_hat
        grad_xy[k + 1] +=  w_tangent * vn * n_hat

    return grad_xy, grad_beta

def delta_beta_limit_grad(beta, delta_beta_max):
    n = len(beta)
    grad = np.zeros(n)

    for k in range(n - 1):
        d = wrap_to_pi(beta[k+1] - beta[k])

        abs_d = abs(d)

        if abs_d > delta_beta_max:
            g = 2.0 * (abs_d - delta_beta_max) * np.sign(d)

            grad[k]     -= g
            grad[k + 1] += g

    return grad

# =========================================================
# MAIN
# =========================================================
def main():

    resolution = 0.05
    origin_xy = (0.0, 0.0)

    obstacle_mask = load_occupancy_from_png(
        # "l_shape_obstacle_45deg.png",
        "l_shape_obstacle_30deg_longwall.png",
        obstacle_is_dark=True,
        thresh=200,
    )
    sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=5.0)

    robot = MowerFrontModel(circle_radius=0.05)

    d_safe = 0.02
    d_edge = d_safe + 0.02

    deck_length = 0.40
    dilate_radius_px = deck_length/2 / resolution

    sample_step_m = 0.05
    sample_step_px = sample_step_m / resolution

    weights = {
        "obstacle": 1.0,
        "edge": 0.0,
        "smooth": 0.01,
        "h": 1.0,
        "h_beta": 1000,
        "tangent": 0.0,
        "delta_beta_limit": 10,
    }

    iters = 250
    history_stride = 5
    alpha = 0.05

    contour_px = extract_largest_contour_px(obstacle_mask)
    dilated_px = dilate_contour_px(contour_px, dilate_radius_px, join_type="round")
    traj_px = resample_closed_contour_uniform_px(dilated_px, sample_step_px)
    traj_px = np.vstack([traj_px, traj_px[0]])

    traj_xy = traj_px * resolution
    beta = init_headings_from_path(traj_xy)
    init_beta = beta.copy()
    init_waypoints = traj_xy.copy()

    traj_initial = traj_xy.copy()
    traj_history = [traj_xy.copy()]

    grad_history = {
        "obstacle": [],
        "edge": [],
        "tangent": [],
        "smooth": [],
        "h": [],
        "h_beta": [],
        "delta_beta_limit": [],
        
    }

    for k in range(iters):

        grad_safe_xy, grad_safe_beta = obstacle_safety_grad(
            traj_xy, beta, robot, sdf,
            origin_xy=origin_xy,
            resolution=resolution,
            d_safe=d_safe,
        )

        grad_edge_xy, grad_edge_beta = edge_band_grad(
            traj_xy, beta, robot, sdf,
            origin_xy=origin_xy,
            resolution=resolution,
            d_edge=d_edge,
        )

        grad_tan_xy, grad_tan_beta = tangent_motion_grad(
            traj_xy, beta, robot, sdf,
            origin_xy=origin_xy,
            resolution=resolution,
            w_tangent=weights["tangent"],
        )

        grad_smooth_xy = smoothness_grad(traj_xy)
        grad_h_xy, grad_h_beta = h_grad(traj_xy, beta)

        grad_delta_beta_limit = delta_beta_limit_grad(beta, delta_beta_max=np.radians(30))


        grad_history["obstacle"].append(np.linalg.norm(grad_safe_xy))
        grad_history["edge"].append(np.linalg.norm(grad_edge_xy))
        grad_history["tangent"].append(np.linalg.norm(grad_tan_xy))
        grad_history["smooth"].append(np.linalg.norm(grad_smooth_xy))
        grad_history["h"].append(np.linalg.norm(grad_h_xy))
        grad_history["h_beta"].append(grad_h_beta)
        grad_history["delta_beta_limit"].append(grad_delta_beta_limit)

        if k % 10 == 0:
            print("Iter: ", k)  
            # print("obstacle grad: ",grad_history["obstacle"][-1])
            # print("edge grad: ",grad_history["edge"][-1])
            # print("tangent grad: ",grad_history["tangent"][-1])
            # print("smooth grad: ",grad_history["smooth"][-1])
            # print("h grad: ",grad_history["h"][-1])

            # print("Max h grad xy: ", np.max(np.abs(grad_h_xy)))
            print("grad h: " , grad_h_xy[102:112])
            print("grad h beta: ", grad_h_beta[102:112])
            print("beta: ", beta[102:112])

            print("delta beta limit grad: ", grad_delta_beta_limit[102:112])

        grad_xy = (
            weights["obstacle"] * grad_safe_xy
            + weights["edge"] * grad_edge_xy
            + weights["tangent"] * grad_tan_xy
            + weights["smooth"] * grad_smooth_xy
            + weights["h"] * grad_h_xy
        )

        grad_beta = (
            weights["obstacle"] * grad_safe_beta
            + weights["edge"] * grad_edge_beta
            + weights["h_beta"]  * grad_h_beta
            + weights["delta_beta_limit"] * grad_delta_beta_limit
        )

        grad_xy[0] = grad_xy[-1] = 0
        grad_beta[0] = grad_beta[-1] = 0

        traj_xy -= alpha * grad_xy
        beta = wrap_to_pi(beta - alpha * grad_beta)

        if (k + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    sdf_vis = sdf.copy()
    sdf_vis[obstacle_mask] = -1.0
    ax.imshow(sdf_vis, cmap="RdYlGn", vmin=-1.0, vmax=4.0, origin="lower")

    cpts = contour_px.reshape(-1, 2)
    dpts = dilated_px.reshape(-1, 2)
    ax.plot(cpts[:, 0], cpts[:, 1], "w--", linewidth=1.0, label="Obstacle contour")
    ax.plot(dpts[:, 0], dpts[:, 1], "y-", linewidth=1.0, label="Dilated init contour")

    ax.plot(traj_initial[:, 0] / resolution, traj_initial[:, 1] / resolution,
            linestyle="--", color="black", linewidth=2, label="Initial")

    for tr in traj_history[1:-1]:
        ax.plot(tr[:, 0] / resolution, tr[:, 1] / resolution, color="black", alpha=0.25)

    traj_final = traj_history[-1]
    ax.plot(traj_final[:, 0] / resolution, traj_final[:, 1] / resolution,
            color="blue", linewidth=2.5, label="Final")

    ax.scatter(init_waypoints[:, 0] / resolution, init_waypoints[:, 1] / resolution,
               color="green", s=14, alpha=0.95, zorder=7, label="Final waypoints")
    ax.scatter(traj_final[:, 0] / resolution, traj_final[:, 1] / resolution,
               color="red", s=14, alpha=0.95, zorder=7, label="Final waypoints")

    # draw mower primitives at final
    for i in range(0, len(traj_final), max(1, len(traj_final)//80)):
        state = (float(traj_final[i, 0]), float(traj_final[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)
        for pr in prims:
            c = pr["center_w"]
            r = pr["radius"]
            tag = pr["tag"]
            ec = "cyan" if tag == "collision" else "magenta"
            ax.add_patch(Circle((c[0] / resolution, c[1] / resolution),
                               radius=r / resolution, fill=False,
                               edgecolor=ec, linewidth=0.7, alpha=0.8))


    # Waypoint numbers
    for i in range(len(traj_initial)):
        ax.text(traj_initial[i,0]/resolution,
                traj_initial[i,1]/resolution,
                str(i), color="black", fontsize=20)

    for i in range(len(traj_final)):
        ax.text(traj_final[i,0]/resolution,
                traj_final[i,1]/resolution,
                str(i), color="blue", fontsize=20)

    # Quiver headings
    arrow_scale = 0.05

    ax.quiver(traj_initial[:,0]/resolution,
              traj_initial[:,1]/resolution,
              arrow_scale*np.cos(init_beta)/resolution,
              arrow_scale*np.sin(init_beta)/resolution,
              color="black", scale=5.0, scale_units="xy")

    ax.quiver(traj_final[:,0]/resolution,
              traj_final[:,1]/resolution,
              arrow_scale*np.cos(beta)/resolution,
              arrow_scale*np.sin(beta)/resolution,
              color="red", scale=5.0, scale_units="xy")


    ax.set_title("sim8: Front Mower — ESDF Edge-Band + Safety + Tangent")
    ax.axis("off")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    np.savez(
    "optimized_traj.npz",
    traj_xy=traj_final,
    beta=beta,
    resolution=resolution,
    origin_xy=np.array(origin_xy),
    )
    print("Saved optimized trajectory to optimized_traj.npz")

if __name__ == "__main__":
    main()
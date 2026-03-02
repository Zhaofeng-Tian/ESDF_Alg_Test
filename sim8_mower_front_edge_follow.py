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
# Costs / Gradients
# =========================================================
def smoothness_grad(traj_xy):
    grad = np.zeros_like(traj_xy)
    grad[1:-1] = 2 * traj_xy[1:-1] - traj_xy[:-2] - traj_xy[2:]
    return grad


def h_grad(traj_xy, beta):
    """
    Same discrete nonholonomic term as your sim8:
      h_k = (cos b_k + cos b_{k+1}) * dy - (sin b_k + sin b_{k+1}) * dx
    """
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


def obstacle_band_and_safety_grad(
    traj_xy,
    beta,
    robot: MowerFrontModel,
    sdf,
    *,
    origin_xy,
    resolution,
    # band objective parameters
    d_edge: float,
    d_safe: float,
    # tangent objective
    w_tangent: float,
):
    """
    ESDF-based objectives for "cut the fence edge as much as possible but don't hit it".

    For each primitive circle, compute d_eff = d(center) - radius.

    - Safety hinge (applies to ALL primitives):
        0.5 * max(0, d_safe - d_eff)^2

    - Edge-band attraction (applies to WORKING/deck primitives):
        0.5 * (d_eff - d_edge)^2

    - Tangential motion encouragement (applies to robot centerline):
        0.5 * sum_k ( (v_k · n_k)^2 ), with n_k from ESDF normal at deck-side representative point
      This discourages moving "into/out of" the fence and encourages along-contour motion.
    """
    n = traj_xy.shape[0]
    grad_xy = np.zeros_like(traj_xy)
    grad_beta = np.zeros(n)

    # ---------- safety + edge band (primitive sum) ----------
    for i in range(n):
        state = (float(traj_xy[i, 0]), float(traj_xy[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)

        for pr in prims:
            c = pr["center_w"]
            r = pr["radius"]
            tag = pr["tag"]
            J_theta = pr["J_theta"]  # (2,)

            d, g = sdf_query(
                sdf,
                float(c[0]),
                float(c[1]),
                origin_xy=origin_xy,
                resolution=resolution,
            )
            # g is ∇d w.r.t center position
            d_eff = d - r

            # ---- safety hinge ----
            if d_eff < d_safe:
                rs = (d_eff - d_safe)  # negative inside unsafe
                # d_eff depends on center only, radius constant
                grad_xy[i] += rs * g
                grad_beta[i] += rs * g.dot(J_theta)

            # ---- edge-band attraction on deck only ----
            if tag == "working":
                re = (d_eff - d_edge)
                grad_xy[i] += re * g
                grad_beta[i] += re * g.dot(J_theta)

    # ---------- tangential encouragement on centerline ----------
    # Use ESDF normal at a representative point: we pick the first "working" primitive in local list.
    # This is a heuristic but works well; you can also average normals across all deck circles.
    if w_tangent != 0.0:
        for k in range(n - 1):
            # representative normal at waypoint k from deck-side
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
                sdf, float(c[0]), float(c[1]),
                origin_xy=origin_xy, resolution=resolution
            )
            ng = np.linalg.norm(g) + 1e-12
            n_hat = g / ng  # outward normal (away from obstacle)

            v = traj_xy[k + 1] - traj_xy[k]
            vn = float(v.dot(n_hat))  # normal component
            # cost 0.5 * vn^2 ; gradient wrt traj points:
            # d/dp_k:  -vn * n_hat
            # d/dp_{k+1}: +vn * n_hat
            grad_xy[k]     += -w_tangent * vn * n_hat
            grad_xy[k + 1] +=  w_tangent * vn * n_hat

    return grad_xy, grad_beta


# =========================================================
# MAIN
# =========================================================
def main():
    # -----------------------------
    # Map / ESDF params
    # -----------------------------
    resolution = 0.05
    origin_xy = (0.0, 0.0)

    obstacle_mask = load_occupancy_from_png(
        "l_shape_obstacle_45deg.png",
        obstacle_is_dark=True,
        thresh=200,
    )
    sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=5.0)

    # -----------------------------
    # Robot model (front mower)
    # -----------------------------
    robot = MowerFrontModel(circle_radius=0.05)

    # -----------------------------
    # Objective parameters (easy to find)
    # -----------------------------
    # Safety: minimum allowed clearance for ANY circle after subtracting its radius.
    d_safe = 0.02  # meters (clearance beyond touching)

    # Edge band: where we WANT the deck circles to sit (after subtracting their radius).
    d_edge = d_safe + 0.02

    # Dilation for initializing a path around the obstacle:
    # Per your note, use the mower deck WIDTH (0.10 m).
    deck_width = 0.10
    deck_length = 0.40
    dilate_radius_px = deck_length/2 / resolution

    # Sampling along contour
    sample_step_m = 0.05
    sample_step_px = sample_step_m / resolution

    # Optimization weights (easy to find)
    weights = {
        "obs": 1.0,        # safety + edge band terms (primitive sum)
        "smooth": 0.01,     # smoothness regularizer
        "h": 0.0,          # nonholonomic regularizer
        "tangent": 0.0,    # encourage tangential motion along fence
    }

    # Optimization loop
    iters = 10
    history_stride = 5
    alpha = 0.05

    # -----------------------------
    # Init path: contour -> dilate -> resample
    # -----------------------------
    contour_px = extract_largest_contour_px(obstacle_mask)
    dilated_px = dilate_contour_px(contour_px, dilate_radius_px, join_type="round")
    traj_px = resample_closed_contour_uniform_px(dilated_px, sample_step_px)
    traj_px = np.vstack([traj_px, traj_px[0]])  # close loop

    traj_xy = traj_px * resolution
    beta = init_headings_from_path(traj_xy)
    init_beta = beta.copy()
    init_waypoints = traj_xy.copy()

    traj_initial = traj_xy.copy()
    traj_history = [traj_xy.copy()]

    for k in range(iters):
        grad_obs_xy, grad_obs_beta = obstacle_band_and_safety_grad(
            traj_xy,
            beta,
            robot,
            sdf,
            origin_xy=origin_xy,
            resolution=resolution,
            d_edge=d_edge,
            d_safe=d_safe,
            w_tangent=weights["tangent"],
        )

        grad_smooth_xy = smoothness_grad(traj_xy)
        grad_h_xy, grad_h_beta = h_grad(traj_xy, beta)

        grad_xy = (
            weights["obs"] * grad_obs_xy
            + weights["smooth"] * grad_smooth_xy
            + weights["h"] * grad_h_xy
        )
        grad_beta = (
            weights["obs"] * grad_obs_beta
            + weights["h"] * 100.0 * grad_h_beta
        )

        # Keep loop seam fixed
        grad_xy[0] = 0
        grad_xy[-1] = 0
        grad_beta[0] = 0
        grad_beta[-1] = 0

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
              color="black", scale=1, scale_units="xy")

    ax.quiver(traj_final[:,0]/resolution,
              traj_final[:,1]/resolution,
              arrow_scale*np.cos(beta)/resolution,
              arrow_scale*np.sin(beta)/resolution,
              color="blue", scale=1, scale_units="xy")


    ax.set_title("sim8: Front Mower — ESDF Edge-Band + Safety + Tangent")
    ax.axis("off")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

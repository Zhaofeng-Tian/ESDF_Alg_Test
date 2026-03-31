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
# Variable indexing for TEB-style state
# poses: x[0:n], y[0:n], beta[0:n], z_dt[0:n-1]
# dt = dt_min + softplus(z_dt)
# =========================================================
def idx_x(i, n):
    return i


def idx_y(i, n):
    return n + i


def idx_b(i, n):
    return 2 * n + i


def idx_dt(i, n):
    return 3 * n + i


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dt_from_z(z_dt, dt_min):
    return dt_min + softplus(z_dt)


def z_from_dt(dt, dt_min):
    y = np.maximum(np.asarray(dt, dtype=float) - dt_min, 1e-6)
    return np.log(np.expm1(y))


def safe_unit(v):
    l = float(np.linalg.norm(v))
    if l <= 1e-12:
        return np.zeros_like(v), 1e-12
    return v / l, l


def interp_angle(a0, a1, t):
    return wrap_to_pi(a0 + t * wrap_to_pi(a1 - a0))


def resize_timed_elastic_band(
    traj_xy,
    beta,
    z_dt,
    *,
    dt_min,
    dt_ref=0.3,
    dt_hyst=0.28,
    max_passes=2,
    min_samples=8,
    max_samples=2000,
):
    """TEB-style online band resizing.

    Insert a new pose between s_k and s_{k+1} if dt_k > dt_ref + dt_hyst.
    Remove pose s_{k+1} if dt_k < dt_ref - dt_hyst and s_{k+1} is interior.
    """
    traj_xy = np.asarray(traj_xy, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy()
    dt = dt_from_z(np.asarray(z_dt, dtype=float), dt_min).copy()

    changed_any = False
    for _ in range(max_passes):
        changed = False

        i = 0
        while i < len(dt) and len(traj_xy) < max_samples:
            if dt[i] > dt_ref + dt_hyst:
                p_mid = 0.5 * (traj_xy[i] + traj_xy[i + 1])
                b_mid = interp_angle(beta[i], beta[i + 1], 0.5)
                dt_half = 0.5 * dt[i]

                traj_xy = np.insert(traj_xy, i + 1, p_mid, axis=0)
                beta = np.insert(beta, i + 1, b_mid)
                dt[i] = dt_half
                dt = np.insert(dt, i + 1, dt_half)

                changed = True
                changed_any = True
                i += 2
            else:
                i += 1

        i = 0
        min_pose_count = max(int(min_samples), 3)
        lower = max(dt_ref - dt_hyst, dt_min + 1e-6)
        while i < len(dt) - 1 and len(traj_xy) > min_pose_count:
            pose_to_remove = i + 1
            if dt[i] < lower and 0 < pose_to_remove < len(traj_xy) - 1:
                traj_xy = np.delete(traj_xy, pose_to_remove, axis=0)
                beta = np.delete(beta, pose_to_remove)
                dt[i] = dt[i] + dt[i + 1]
                dt = np.delete(dt, i + 1)
                changed = True
                changed_any = True
                continue
            i += 1

        if not changed:
            break

    z_dt = z_from_dt(dt, dt_min)
    return traj_xy, beta, z_dt, dt, changed_any


# =========================================================
# ESDF cache: exactly one sdf_query pass per outer iteration
# =========================================================
def build_esdf_cache(traj_xy, beta, robot, sdf, *, origin_xy, resolution):
    cache = []
    for i in range(traj_xy.shape[0]):
        state = (float(traj_xy[i, 0]), float(traj_xy[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)
        prim_cache = []
        for pr in prims:
            c = pr["center_w"]
            d, g = sdf_query(
                sdf,
                float(c[0]),
                float(c[1]),
                origin_xy=origin_xy,
                resolution=resolution,
            )
            prim_cache.append(
                {
                    "tag": pr["tag"],
                    "center_w": np.asarray(pr["center_w"], dtype=float),
                    "radius": float(pr["radius"]),
                    "J_theta": np.asarray(pr["J_theta"], dtype=float),
                    "d": float(d),
                    "g": np.asarray(g, dtype=float),
                }
            )
        cache.append(prim_cache)
    return cache


# =========================================================
# TEB-style residual + analytic Jacobian builder
# =========================================================
def build_residual_and_jacobian_teb(
    traj_xy,
    beta,
    z_dt,
    esdf_cache,
    *,
    weights,
    d_safe,
    d_edge,
    dt_min,
    teb_kappa=100.0,
    v_max=0.5,
    omega_max=1.0,
    a_max=0.8,
    alpha_max=2.0,
    min_turning_radius=1.0,
):
    n = traj_xy.shape[0]
    m = n - 1
    nv = 4 * n - 1
    rows = []
    jac_rows = []
    tags = []

    dt = dt_from_z(z_dt, dt_min)
    ddt_dz = sigmoid(z_dt)

    def add_row(residual, entries, tag):
        row = np.zeros(nv, dtype=float)
        for col, val in entries:
            row[col] += val
        rows.append(float(residual))
        jac_rows.append(row)
        tags.append(tag)

    def add_hinge_row(value, grad_entries, limit, sqrt_w, tag):
        excess = value - limit
        if excess > 0.0 and sqrt_w > 0.0:
            add_row(sqrt_w * excess, [(col, sqrt_w * g) for col, g in grad_entries], tag)

    sqrt_w_obs = np.sqrt(max(weights.get("obstacle", 0.0), 0.0))
    sqrt_w_edge = np.sqrt(max(weights.get("edge", 0.0), 0.0))
    sqrt_w_smooth = np.sqrt(max(weights.get("smooth", 0.0), 0.0))
    sqrt_w_h = np.sqrt(max(weights.get("h", 0.0), 0.0))
    sqrt_w_time = np.sqrt(max(weights.get("time", 0.0), 0.0))
    sqrt_w_v = np.sqrt(max(weights.get("velocity", 0.0), 0.0))
    sqrt_w_omega = np.sqrt(max(weights.get("omega", 0.0), 0.0))
    sqrt_w_a = np.sqrt(max(weights.get("accel", 0.0), 0.0))
    sqrt_w_alpha = np.sqrt(max(weights.get("alpha", 0.0), 0.0))
    sqrt_w_turn = np.sqrt(max(weights.get("turning_radius", 0.0), 0.0))
    sqrt_w_dt_eq = np.sqrt(max(weights.get("dt_equalize", 0.0), 0.0))

    # 1) Safety term: active-set hinge residuals on all circle primitives
    if sqrt_w_obs > 0.0:
        for i in range(n):
            for pr in esdf_cache[i]:
                d_eff = pr["d"] - pr["radius"]
                if d_eff < d_safe:
                    g = pr["g"]
                    jt = pr["J_theta"]
                    add_row(
                        sqrt_w_obs * (d_eff - d_safe),
                        [
                            (idx_x(i, n), sqrt_w_obs * g[0]),
                            (idx_y(i, n), sqrt_w_obs * g[1]),
                            (idx_b(i, n), sqrt_w_obs * float(g.dot(jt))),
                        ],
                        "obstacle",
                    )

    # 2) Keep edge term for the working primitive
    if sqrt_w_edge > 0.0:
        for i in range(n):
            work = None
            for pr in esdf_cache[i]:
                if pr["tag"] == "working":
                    work = pr
                    break
            if work is None:
                continue
            d_eff = work["d"] - work["radius"]
            g = work["g"]
            jt = work["J_theta"]
            add_row(
                sqrt_w_edge * (d_eff - d_edge),
                [
                    (idx_x(i, n), sqrt_w_edge * g[0]),
                    (idx_y(i, n), sqrt_w_edge * g[1]),
                    (idx_b(i, n), sqrt_w_edge * float(g.dot(jt))),
                ],
                "edge",
            )

    # 3) Weak smoothness regularizer on positions only
    if sqrt_w_smooth > 0.0:
        for i in range(1, n - 1):
            rx = sqrt_w_smooth * (2.0 * traj_xy[i, 0] - traj_xy[i - 1, 0] - traj_xy[i + 1, 0])
            ry = sqrt_w_smooth * (2.0 * traj_xy[i, 1] - traj_xy[i - 1, 1] - traj_xy[i + 1, 1])
            add_row(
                rx,
                [
                    (idx_x(i - 1, n), -sqrt_w_smooth),
                    (idx_x(i, n), 2.0 * sqrt_w_smooth),
                    (idx_x(i + 1, n), -sqrt_w_smooth),
                ],
                "smooth_x",
            )
            add_row(
                ry,
                [
                    (idx_y(i - 1, n), -sqrt_w_smooth),
                    (idx_y(i, n), 2.0 * sqrt_w_smooth),
                    (idx_y(i + 1, n), -sqrt_w_smooth),
                ],
                "smooth_y",
            )

    # Per-segment quantities used by multiple terms
    v_vals = np.zeros(m, dtype=float)
    w_vals = np.zeros(m, dtype=float)
    dv_entries = []
    dw_entries = []

    for k in range(m):
        seg = traj_xy[k + 1] - traj_xy[k]
        u, seg_len = safe_unit(seg)
        dx, dy = float(seg[0]), float(seg[1])
        cb = float(np.cos(beta[k]))
        sb = float(np.sin(beta[k]))
        proj = cb * dx + sb * dy
        kp = teb_kappa * proj
        gamma = kp / (1.0 + abs(kp))
        dgamma_dproj = teb_kappa / (1.0 + abs(kp)) ** 2

        dt_k = float(dt[k])
        inv_dt = 1.0 / max(dt_k, 1e-9)

        dl = {
            idx_x(k, n): -u[0],
            idx_y(k, n): -u[1],
            idx_x(k + 1, n): u[0],
            idx_y(k + 1, n): u[1],
        }
        dproj = {
            idx_x(k, n): -cb,
            idx_y(k, n): -sb,
            idx_x(k + 1, n): cb,
            idx_y(k + 1, n): sb,
            idx_b(k, n): -sb * dx + cb * dy,
        }

        dv_map = {}
        for col, val in dl.items():
            dv_map[col] = dv_map.get(col, 0.0) + gamma * inv_dt * val
        for col, val in dproj.items():
            dv_map[col] = dv_map.get(col, 0.0) + (seg_len * inv_dt * dgamma_dproj) * val
        dv_map[idx_dt(k, n)] = dv_map.get(idx_dt(k, n), 0.0) - (seg_len * gamma) / (dt_k * dt_k) * ddt_dz[k]

        v_k = seg_len * gamma * inv_dt
        v_vals[k] = v_k
        dv_entries.append(list(dv_map.items()))

        dtheta = wrap_to_pi(beta[k + 1] - beta[k])
        w_k = dtheta * inv_dt
        w_vals[k] = w_k
        dw_entries.append(
            [
                (idx_b(k, n), -inv_dt),
                (idx_b(k + 1, n), inv_dt),
                (idx_dt(k, n), -(dtheta / (dt_k * dt_k)) * ddt_dz[k]),
            ]
        )

        # 4) Nonholonomic residual (keep strongly weighted)
        if sqrt_w_h > 0.0:
            cbk = np.cos(beta[k])
            sbk = np.sin(beta[k])
            cbk1 = np.cos(beta[k + 1])
            sbk1 = np.sin(beta[k + 1])
            h = (cbk + cbk1) * dy - (sbk + sbk1) * dx
            add_row(
                sqrt_w_h * h,
                [
                    (idx_x(k, n), sqrt_w_h * (sbk + sbk1)),
                    (idx_y(k, n), -sqrt_w_h * (cbk + cbk1)),
                    (idx_x(k + 1, n), -sqrt_w_h * (sbk + sbk1)),
                    (idx_y(k + 1, n), sqrt_w_h * (cbk + cbk1)),
                    (idx_b(k, n), sqrt_w_h * ((-sbk) * dy - cbk * dx)),
                    (idx_b(k + 1, n), sqrt_w_h * ((-sbk1) * dy - cbk1 * dx)),
                ],
                "nonholonomic",
            )

        # 5) Minimum-time residual
        if sqrt_w_time > 0.0:
            add_row(
                sqrt_w_time * dt_k,
                [(idx_dt(k, n), sqrt_w_time * ddt_dz[k])],
                "time",
            )

        # 6) Translational velocity hinge
        if sqrt_w_v > 0.0:
            sign_v = 1.0 if v_k >= 0.0 else -1.0
            add_hinge_row(abs(v_k), [(c, sign_v * g) for c, g in dv_entries[-1]], v_max, sqrt_w_v, "velocity")

        # 7) Angular velocity hinge
        if sqrt_w_omega > 0.0:
            sign_w = 1.0 if w_k >= 0.0 else -1.0
            add_hinge_row(abs(w_k), [(c, sign_w * g) for c, g in dw_entries[-1]], omega_max, sqrt_w_omega, "omega")

        # 8) Minimum turning radius hinge via curvature proxy |dtheta| <= seg_len / r_min
        if sqrt_w_turn > 0.0 and min_turning_radius > 1e-9:
            abs_db = abs(dtheta)
            sign_db = 1.0 if dtheta >= 0.0 else -1.0
            val = abs_db - seg_len / min_turning_radius
            if val > 0.0:
                entries = [
                    (idx_b(k, n), -sign_db),
                    (idx_b(k + 1, n), sign_db),
                    (idx_x(k, n), u[0] / min_turning_radius),
                    (idx_y(k, n), u[1] / min_turning_radius),
                    (idx_x(k + 1, n), -u[0] / min_turning_radius),
                    (idx_y(k + 1, n), -u[1] / min_turning_radius),
                ]
                add_row(sqrt_w_turn * val, [(c, sqrt_w_turn * g) for c, g in entries], "turning_radius")

    # 9) Translational acceleration hinge
    if sqrt_w_a > 0.0:
        for k in range(m - 1):
            denom = max(dt[k] + dt[k + 1], 1e-9)
            a_k = 2.0 * (v_vals[k + 1] - v_vals[k]) / denom
            sign_a = 1.0 if a_k >= 0.0 else -1.0
            abs_a = abs(a_k)
            if abs_a > a_max:
                scale = 2.0 / denom
                grad = {}
                for c, g in dv_entries[k + 1]:
                    grad[c] = grad.get(c, 0.0) + scale * g
                for c, g in dv_entries[k]:
                    grad[c] = grad.get(c, 0.0) - scale * g
                coeff_denom = -2.0 * (v_vals[k + 1] - v_vals[k]) / (denom * denom)
                grad[idx_dt(k, n)] = grad.get(idx_dt(k, n), 0.0) + coeff_denom * ddt_dz[k]
                grad[idx_dt(k + 1, n)] = grad.get(idx_dt(k + 1, n), 0.0) + coeff_denom * ddt_dz[k + 1]
                add_row(
                    sqrt_w_a * (abs_a - a_max),
                    [(c, sqrt_w_a * sign_a * g) for c, g in grad.items()],
                    "accel",
                )

    # 10) Angular acceleration hinge
    if sqrt_w_alpha > 0.0:
        for k in range(m - 1):
            denom = max(dt[k] + dt[k + 1], 1e-9)
            alpha_k = 2.0 * (w_vals[k + 1] - w_vals[k]) / denom
            sign_alpha = 1.0 if alpha_k >= 0.0 else -1.0
            abs_alpha = abs(alpha_k)
            if abs_alpha > alpha_max:
                scale = 2.0 / denom
                grad = {}
                for c, g in dw_entries[k + 1]:
                    grad[c] = grad.get(c, 0.0) + scale * g
                for c, g in dw_entries[k]:
                    grad[c] = grad.get(c, 0.0) - scale * g
                coeff_denom = -2.0 * (w_vals[k + 1] - w_vals[k]) / (denom * denom)
                grad[idx_dt(k, n)] = grad.get(idx_dt(k, n), 0.0) + coeff_denom * ddt_dz[k]
                grad[idx_dt(k + 1, n)] = grad.get(idx_dt(k + 1, n), 0.0) + coeff_denom * ddt_dz[k + 1]
                add_row(
                    sqrt_w_alpha * (abs_alpha - alpha_max),
                    [(c, sqrt_w_alpha * sign_alpha * g) for c, g in grad.items()],
                    "alpha",
                )

    # 11) Optional weak temporal equalization / regularization
    if sqrt_w_dt_eq > 0.0:
        for k in range(m - 1):
            add_row(
                sqrt_w_dt_eq * (dt[k + 1] - dt[k]),
                [
                    (idx_dt(k, n), -sqrt_w_dt_eq * ddt_dz[k]),
                    (idx_dt(k + 1, n), sqrt_w_dt_eq * ddt_dz[k + 1]),
                ],
                "dt_equalize",
            )

    if not rows:
        return np.zeros(0, dtype=float), np.zeros((0, nv), dtype=float), tags, dt

    r = np.asarray(rows, dtype=float)
    J = np.vstack(jac_rows)
    return r, J, tags, dt


# =========================================================
# LM optimizer with one ESDF pass per outer iteration
# =========================================================
def lm_optimize_teb(
    traj_xy,
    beta,
    robot,
    sdf,
    *,
    origin_xy,
    resolution,
    d_safe,
    d_edge,
    weights,
    dt_init,
    dt_min=0.03,
    dt_ref=0.30,
    dt_hyst=0.28,
    resize_max_passes=2,
    teb_kappa=100.0,
    v_max=0.5,
    omega_max=1.0,
    a_max=0.8,
    alpha_max=2.0,
    min_turning_radius=0.0,
    iters=100,
    history_stride=5,
    lambda0=1e-2,
    step_clip_xy=0.05,
    step_clip_beta=np.radians(10.0),
    step_clip_dt_z=0.35,
):
    traj_xy = traj_xy.copy()
    beta = beta.copy()
    m = traj_xy.shape[0] - 1
    z_dt = z_from_dt(np.full(m, dt_init, dtype=float), dt_min)

    traj_history = [traj_xy.copy()]
    beta_history = [beta.copy()]
    dt_history = [dt_from_z(z_dt, dt_min).copy()]
    obj_history = []
    lambda_lm = float(lambda0)

    fixed_xy0 = traj_xy[0].copy()
    fixed_xyN = traj_xy[-1].copy()
    fixed_b0 = float(beta[0])
    fixed_bN = float(beta[-1])

    for it in range(iters):
        traj_xy, beta, z_dt, dt_resized, resized = resize_timed_elastic_band(
            traj_xy,
            beta,
            z_dt,
            dt_min=dt_min,
            dt_ref=dt_ref,
            dt_hyst=dt_hyst,
            max_passes=resize_max_passes,
            min_samples=8,
        )

        traj_xy[0] = fixed_xy0
        traj_xy[-1] = fixed_xyN
        beta[0] = fixed_b0
        beta[-1] = fixed_bN

        n = traj_xy.shape[0]
        nv = 4 * n - 1
        free_mask = np.ones(nv, dtype=bool)
        for i in [0, n - 1]:
            free_mask[idx_x(i, n)] = False
            free_mask[idx_y(i, n)] = False
            free_mask[idx_b(i, n)] = False

        esdf_cache = build_esdf_cache(
            traj_xy,
            beta,
            robot,
            sdf,
            origin_xy=origin_xy,
            resolution=resolution,
        )

        r, J, tags, dt = build_residual_and_jacobian_teb(
            traj_xy,
            beta,
            z_dt,
            esdf_cache,
            weights=weights,
            d_safe=d_safe,
            d_edge=d_edge,
            dt_min=dt_min,
            teb_kappa=teb_kappa,
            v_max=v_max,
            omega_max=omega_max,
            a_max=a_max,
            alpha_max=alpha_max,
            min_turning_radius=min_turning_radius,
        )

        obj = 0.5 * float(r.dot(r)) if r.size else 0.0
        obj_history.append(obj)

        if it % 10 == 0:
            counts = {}
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
            print(
                f"Iter {it:4d} | obj={obj:.6e} | lambda={lambda_lm:.3e} | "
                f"n={n} | resized={resized} | residuals={len(tags)} | active={counts} | dt=[{dt.min():.3f},{dt.max():.3f}]"
            )

        if r.size == 0:
            if (it + 1) % history_stride == 0:
                traj_history.append(traj_xy.copy())
                beta_history.append(beta.copy())
                dt_history.append(dt.copy())
            continue

        Jf = J[:, free_mask]
        H = Jf.T @ Jf
        g = Jf.T @ r

        damp = np.diag(np.diag(H)) + 1e-9 * np.eye(H.shape[0])
        A = H + lambda_lm * damp

        solved = False
        for _ in range(6):
            try:
                delta_free = -np.linalg.solve(A, g)
                solved = True
                break
            except np.linalg.LinAlgError:
                lambda_lm *= 10.0
                A = H + lambda_lm * damp

        if not solved:
            print("LM solve failed repeatedly; stopping.")
            break

        delta = np.zeros(nv, dtype=float)
        delta[free_mask] = delta_free

        dx_step = delta[0:n]
        dy_step = delta[n : 2 * n]
        db_step = delta[2 * n : 3 * n]
        dzdt_step = delta[3 * n :]

        step_xy_norm = np.sqrt(dx_step * dx_step + dy_step * dy_step)
        scale_xy = np.ones_like(step_xy_norm)
        mask_big = step_xy_norm > step_clip_xy
        scale_xy[mask_big] = step_clip_xy / np.maximum(step_xy_norm[mask_big], 1e-12)
        dx_step *= scale_xy
        dy_step *= scale_xy
        db_step = np.clip(db_step, -step_clip_beta, step_clip_beta)
        dzdt_step = np.clip(dzdt_step, -step_clip_dt_z, step_clip_dt_z)

        traj_xy[:, 0] += dx_step
        traj_xy[:, 1] += dy_step
        beta = wrap_to_pi(beta + db_step)
        z_dt += dzdt_step

        # Keep endpoints fixed exactly.
        traj_xy[0] = fixed_xy0
        traj_xy[-1] = fixed_xyN
        beta[0] = fixed_b0
        beta[-1] = fixed_bN

        # Mild damping schedule without extra ESDF calls in this iteration.
        lambda_lm = max(lambda_lm * 0.98, 1e-6)

        if (it + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())
            beta_history.append(beta.copy())
            dt_history.append(dt_from_z(z_dt, dt_min).copy())

    return traj_xy, beta, dt_from_z(z_dt, dt_min), traj_history, beta_history, dt_history, obj_history


# =========================================================
# MAIN
# =========================================================
def main():
    resolution = 0.05
    origin_xy = (0.0, 0.0)

    obstacle_mask = load_occupancy_from_png(
        # "l_shape_obstacle_30deg_longwall.png",
        "l_shape_obstacle_45deg.png",  # --- IGNORE ---
        obstacle_is_dark=True,
        thresh=200,
    )
    sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=5.0)

    robot = MowerFrontModel(circle_radius=0.05)

    d_safe = 0.02
    d_edge = d_safe + 0.02

    deck_length = 0.40
    dilate_radius_px = deck_length / 2 / resolution

    sample_step_m = 0.05
    sample_step_px = sample_step_m / resolution

    # TEB-paper-inspired baseline: unit weights for most terms, strong
    # nonholonomic equality penalty; edge/smooth are custom for coverage.
    weights = {
        "obstacle": 2.0,
        "edge": 100.0,
        "smooth": 1e-4,
        "h": 1000.0,
        "time": 1.0,
        "velocity": 1.0,
        "omega": 1.0,
        "accel": 1.0,
        "alpha": 1.0,
        "turning_radius": 1.0,
        "dt_equalize": 1.0,
        "tangent": 0.0,
    }

    iters =200
    history_stride = 5

    contour_px = extract_largest_contour_px(obstacle_mask)
    dilated_px = dilate_contour_px(contour_px, dilate_radius_px, join_type="round")
    traj_px = resample_closed_contour_uniform_px(dilated_px, sample_step_px)
    traj_px = np.vstack([traj_px, traj_px[0]])

    traj_xy = traj_px * resolution
    beta = init_headings_from_path(traj_xy)
    init_beta = beta.copy()
    init_waypoints = traj_xy.copy()
    traj_initial = traj_xy.copy()

    nominal_speed = 0.25
    dt_init = max(sample_step_m / nominal_speed, 0.10)
    dt_ref = 0.2
    dt_hyst = 0.85 * dt_ref

    traj_final, beta_final, dt_final, traj_history, beta_history, dt_history, obj_history = lm_optimize_teb(
        traj_xy,
        beta,
        robot,
        sdf,
        origin_xy=origin_xy,
        resolution=resolution,
        d_safe=d_safe,
        d_edge=d_edge,
        weights=weights,
        dt_init=dt_init,
        dt_min=0.03,
        dt_ref=dt_ref,
        dt_hyst=dt_hyst,
        resize_max_passes=2,
        teb_kappa=100.0,
        v_max=0.40,
        omega_max=0.8,
        a_max=0.8,
        alpha_max=2.0,
        min_turning_radius=1.0,
        iters=iters,
        history_stride=history_stride,
        lambda0=1e-2,
        step_clip_xy=0.05,
        step_clip_beta=np.radians(8.0),
        step_clip_dt_z=0.35,
    )

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

    ax.plot(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Initial",
    )

    for tr in traj_history[1:-1]:
        ax.plot(tr[:, 0] / resolution, tr[:, 1] / resolution, color="black", alpha=0.25)

    ax.plot(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        linewidth=2.5,
        label="Final",
    )

    ax.scatter(
        init_waypoints[:, 0] / resolution,
        init_waypoints[:, 1] / resolution,
        color="green",
        s=14,
        alpha=0.95,
        zorder=7,
        label="Initial waypoints",
    )
    ax.scatter(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="red",
        s=14,
        alpha=0.95,
        zorder=7,
        label="Final waypoints",
    )

    for i in range(0, len(traj_final), max(1, len(traj_final) // 80)):
        state = (float(traj_final[i, 0]), float(traj_final[i, 1]), float(beta_final[i]))
        prims = robot.world_centers_and_jacobians(state)
        for pr in prims:
            c = pr["center_w"]
            r = pr["radius"]
            tag = pr["tag"]
            ec = "cyan" if tag == "collision" else "magenta"
            ax.add_patch(
                Circle(
                    (c[0] / resolution, c[1] / resolution),
                    radius=r / resolution,
                    fill=False,
                    edgecolor=ec,
                    linewidth=0.7,
                    alpha=0.8,
                )
            )

    arrow_scale = 0.05
    ax.quiver(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        arrow_scale * np.cos(init_beta) / resolution,
        arrow_scale * np.sin(init_beta) / resolution,
        color="black",
        scale=5.0,
        scale_units="xy",
    )

    ax.quiver(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        arrow_scale * np.cos(beta_final) / resolution,
        arrow_scale * np.sin(beta_final) / resolution,
        color="red",
        scale=5.0,
        scale_units="xy",
    )

    ax.set_title("sim14: LM cached-ESDF + TEB-style residuals + resizing")
    ax.axis("off")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    np.savez(
        "optimized_traj_lm_teb_resized.npz",
        traj_xy=traj_final,
        beta=beta_final,
        dt=dt_final,
        traj_history=np.array(traj_history, dtype=object),
        beta_history=np.array(beta_history, dtype=object),
        dt_history=np.array(dt_history, dtype=object),
        obj_history=np.array(obj_history, dtype=float),
        resolution=resolution,
        origin_xy=np.array(origin_xy),
    )
    print("Saved optimized trajectory to optimized_traj_lm_teb_style.npz")
    print(f"Final dt range: [{dt_final.min():.4f}, {dt_final.max():.4f}] s")
    print(f"Approx total time: {dt_final.sum():.4f} s")
    print(f"TEB resizing params: dt_ref={dt_ref:.3f} s, dt_hyst={dt_hyst:.3f} s")


if __name__ == "__main__":
    main()

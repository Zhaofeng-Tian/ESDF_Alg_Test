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
# Variable indexing
# =========================================================
def idx_x(i):
    return 3 * i


def idx_y(i):
    return 3 * i + 1


def idx_b(i):
    return 3 * i + 2


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
# Residual + analytic Jacobian builder
# =========================================================
def build_residual_and_jacobian(
    traj_xy,
    beta,
    esdf_cache,
    *,
    weights,
    d_safe,
    d_edge,
    delta_beta_max,
    segment_length_max=None,
    use_tangent=True,
):
    n = traj_xy.shape[0]
    nv = 3 * n
    rows = []
    jac_rows = []
    tags = []

    def add_row(residual, entries, tag):
        row = np.zeros(nv, dtype=float)
        for col, val in entries:
            row[col] += val
        rows.append(float(residual))
        jac_rows.append(row)
        tags.append(tag)

    sqrt_w_obs = np.sqrt(max(weights.get("obstacle", 0.0), 0.0))
    sqrt_w_edge = np.sqrt(max(weights.get("edge", 0.0), 0.0))
    sqrt_w_smooth = np.sqrt(max(weights.get("smooth", 0.0), 0.0))
    sqrt_w_h = np.sqrt(max(weights.get("h", 0.0), 0.0))
    sqrt_w_tan = np.sqrt(max(weights.get("tangent", 0.0), 0.0))
    sqrt_w_db = np.sqrt(max(weights.get("delta_beta_limit", 0.0), 0.0))
    sqrt_w_segmax = np.sqrt(max(weights.get("segment_length_max", 0.0), 0.0))
    sqrt_w_segeq = np.sqrt(max(weights.get("segment_length_equalize", 0.0), 0.0))

    # 1) Safety term: active-set hinge residuals
    if sqrt_w_obs > 0.0:
        for i in range(n):
            for pr in esdf_cache[i]:
                d_eff = pr["d"] - pr["radius"]
                if d_eff < d_safe:
                    base = sqrt_w_obs * (d_eff - d_safe)
                    g = pr["g"]
                    jt = pr["J_theta"]
                    add_row(
                        base,
                        [
                            (idx_x(i), sqrt_w_obs * g[0]),
                            (idx_y(i), sqrt_w_obs * g[1]),
                            (idx_b(i), sqrt_w_obs * float(g.dot(jt))),
                        ],
                        "obstacle",
                    )

    # 2) Edge band term on working primitive
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
            base = sqrt_w_edge * (d_eff - d_edge)
            g = work["g"]
            jt = work["J_theta"]
            add_row(
                base,
                [
                    (idx_x(i), sqrt_w_edge * g[0]),
                    (idx_y(i), sqrt_w_edge * g[1]),
                    (idx_b(i), sqrt_w_edge * float(g.dot(jt))),
                ],
                "edge",
            )

    # 3) Tangent motion term with frozen ESDF normal per iteration
    if use_tangent and sqrt_w_tan > 0.0:
        for k in range(n - 1):
            work = None
            for pr in esdf_cache[k]:
                if pr["tag"] == "working":
                    work = pr
                    break
            if work is None:
                continue
            g = work["g"]
            n_hat = g / (np.linalg.norm(g) + 1e-12)
            v = traj_xy[k + 1] - traj_xy[k]
            vn = float(v.dot(n_hat))
            add_row(
                sqrt_w_tan * vn,
                [
                    (idx_x(k), -sqrt_w_tan * n_hat[0]),
                    (idx_y(k), -sqrt_w_tan * n_hat[1]),
                    (idx_x(k + 1), sqrt_w_tan * n_hat[0]),
                    (idx_y(k + 1), sqrt_w_tan * n_hat[1]),
                ],
                "tangent",
            )

    # 4) Smoothness term: keep original second-difference form
    if sqrt_w_smooth > 0.0:
        for i in range(1, n - 1):
            rx = sqrt_w_smooth * (2.0 * traj_xy[i, 0] - traj_xy[i - 1, 0] - traj_xy[i + 1, 0])
            ry = sqrt_w_smooth * (2.0 * traj_xy[i, 1] - traj_xy[i - 1, 1] - traj_xy[i + 1, 1])
            add_row(
                rx,
                [
                    (idx_x(i - 1), -sqrt_w_smooth),
                    (idx_x(i), 2.0 * sqrt_w_smooth),
                    (idx_x(i + 1), -sqrt_w_smooth),
                ],
                "smooth_x",
            )
            add_row(
                ry,
                [
                    (idx_y(i - 1), -sqrt_w_smooth),
                    (idx_y(i), 2.0 * sqrt_w_smooth),
                    (idx_y(i + 1), -sqrt_w_smooth),
                ],
                "smooth_y",
            )

    # 5) Nonholonomic residual
    if sqrt_w_h > 0.0:
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
            dh_dbk = (-sbk) * dy - cbk * dx
            dh_dbk1 = (-sbk1) * dy - cbk1 * dx

            add_row(
                sqrt_w_h * h,
                [
                    (idx_x(k), sqrt_w_h * dh_dxk),
                    (idx_y(k), sqrt_w_h * dh_dyk),
                    (idx_x(k + 1), sqrt_w_h * dh_dxk1),
                    (idx_y(k + 1), sqrt_w_h * dh_dyk1),
                    (idx_b(k), sqrt_w_h * dh_dbk),
                    (idx_b(k + 1), sqrt_w_h * dh_dbk1),
                ],
                "nonholonomic",
            )

    # 6) Delta-beta limit hinge residual
    if sqrt_w_db > 0.0:
        for k in range(n - 1):
            d = wrap_to_pi(beta[k + 1] - beta[k])
            abs_d = abs(d)
            if abs_d > delta_beta_max:
                sgn = 1.0 if d >= 0.0 else -1.0
                res = sqrt_w_db * (abs_d - delta_beta_max)
                add_row(
                    res,
                    [
                        (idx_b(k), -sqrt_w_db * sgn),
                        (idx_b(k + 1), sqrt_w_db * sgn),
                    ],
                    "delta_beta_limit",
                )

    # 7) Maximum segment length hinge residual
    if sqrt_w_segmax > 0.0 and segment_length_max is not None:
        for k in range(n - 1):
            seg = traj_xy[k + 1] - traj_xy[k]
            seg_len = float(np.linalg.norm(seg))
            if seg_len > segment_length_max and seg_len > 1e-12:
                u = seg / seg_len
                res = sqrt_w_segmax * (seg_len - segment_length_max)
                add_row(
                    res,
                    [
                        (idx_x(k), -sqrt_w_segmax * u[0]),
                        (idx_y(k), -sqrt_w_segmax * u[1]),
                        (idx_x(k + 1), sqrt_w_segmax * u[0]),
                        (idx_y(k + 1), sqrt_w_segmax * u[1]),
                    ],
                    "segment_length_max",
                )

    # 8) Segment length equalization residual
    if sqrt_w_segeq > 0.0:
        for k in range(n - 2):
            v0 = traj_xy[k + 1] - traj_xy[k]
            v1 = traj_xy[k + 2] - traj_xy[k + 1]
            l0 = float(np.linalg.norm(v0))
            l1 = float(np.linalg.norm(v1))
            if l0 <= 1e-12 or l1 <= 1e-12:
                continue
            u0 = v0 / l0
            u1 = v1 / l1
            res = sqrt_w_segeq * (l1 - l0)
            add_row(
                res,
                [
                    (idx_x(k), sqrt_w_segeq * u0[0]),
                    (idx_y(k), sqrt_w_segeq * u0[1]),
                    (idx_x(k + 1), sqrt_w_segeq * (-u0[0] - u1[0])),
                    (idx_y(k + 1), sqrt_w_segeq * (-u0[1] - u1[1])),
                    (idx_x(k + 2), sqrt_w_segeq * u1[0]),
                    (idx_y(k + 2), sqrt_w_segeq * u1[1]),
                ],
                "segment_length_equalize",
            )

    if not rows:
        return np.zeros(0, dtype=float), np.zeros((0, nv), dtype=float), tags

    r = np.asarray(rows, dtype=float)
    J = np.vstack(jac_rows)
    return r, J, tags


# =========================================================
# LM optimizer with one ESDF pass per outer iteration
# =========================================================
def lm_optimize(
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
    iters=100,
    history_stride=5,
    delta_beta_max=np.radians(30.0),
    segment_length_max=None,
    lambda0=1e-2,
    step_clip_xy=0.05,
    step_clip_beta=np.radians(10.0),
):
    traj_xy = traj_xy.copy()
    beta = beta.copy()
    n = traj_xy.shape[0]
    nv = 3 * n

    free_mask = np.ones(nv, dtype=bool)
    for i in [0, n - 1]:
        free_mask[idx_x(i)] = False
        free_mask[idx_y(i)] = False
        free_mask[idx_b(i)] = False

    traj_history = [traj_xy.copy()]
    beta_history = [beta.copy()]
    obj_history = []
    lambda_lm = float(lambda0)

    for it in range(iters):
        esdf_cache = build_esdf_cache(
            traj_xy,
            beta,
            robot,
            sdf,
            origin_xy=origin_xy,
            resolution=resolution,
        )

        r, J, tags = build_residual_and_jacobian(
            traj_xy,
            beta,
            esdf_cache,
            weights=weights,
            d_safe=d_safe,
            d_edge=d_edge,
            delta_beta_max=delta_beta_max,
            segment_length_max=segment_length_max,
            use_tangent=True,
        )

        obj = 0.5 * float(r.dot(r)) if r.size else 0.0
        obj_history.append(obj)

        if it % 10 == 0:
            counts = {}
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
            print(f"Iter {it:4d} | obj={obj:.6e} | lambda={lambda_lm:.3e} | residuals={len(tags)} | active={counts}")

        if r.size == 0:
            if (it + 1) % history_stride == 0:
                traj_history.append(traj_xy.copy())
                beta_history.append(beta.copy())
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

        dx_step = delta[0::3]
        dy_step = delta[1::3]
        db_step = delta[2::3]

        step_xy_norm = np.sqrt(dx_step * dx_step + dy_step * dy_step)
        scale_xy = np.ones_like(step_xy_norm)
        mask_big = step_xy_norm > step_clip_xy
        scale_xy[mask_big] = step_clip_xy / np.maximum(step_xy_norm[mask_big], 1e-12)
        dx_step *= scale_xy
        dy_step *= scale_xy
        db_step = np.clip(db_step, -step_clip_beta, step_clip_beta)

        traj_xy[:, 0] += dx_step
        traj_xy[:, 1] += dy_step
        beta = wrap_to_pi(beta + db_step)

        # Keep endpoints fixed exactly.
        traj_xy[0] = traj_history[0][0]
        traj_xy[-1] = traj_history[0][-1]
        beta[0] = beta_history[0][0]
        beta[-1] = beta_history[0][-1]

        # Mild damping schedule without extra ESDF calls in this iteration.
        lambda_lm = max(lambda_lm * 0.98, 1e-6)

        if (it + 1) % history_stride == 0:
            traj_history.append(traj_xy.copy())
            beta_history.append(beta.copy())

    return traj_xy, beta, traj_history, beta_history, obj_history


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
    segment_length_max = 1.6 * sample_step_m

    weights = {
        "obstacle": 1.0,
        "edge": 0.1,
        "smooth": 0.001,
        "h": 1.0,
        # LM uses one clean nonholonomic residual weight.
        "h_beta": 1000.0,
        "tangent": 0.0,
        "delta_beta_limit": 10.0,
        "segment_length_max": 20.0,
        "segment_length_equalize": 2.0,
    }

    iters = 250
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

    traj_final, beta_final, traj_history, beta_history, obj_history = lm_optimize(
        traj_xy,
        beta,
        robot,
        sdf,
        origin_xy=origin_xy,
        resolution=resolution,
        d_safe=d_safe,
        d_edge=d_edge,
        weights=weights,
        iters=iters,
        history_stride=history_stride,
        delta_beta_max=np.radians(30.0),
        segment_length_max=segment_length_max,
        lambda0=1e-2,
        step_clip_xy=0.05,
        step_clip_beta=np.radians(8.0),
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

    ax.set_title("sim12: LM cached-ESDF + analytic Jacobian")
    ax.axis("off")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    np.savez(
        "optimized_traj_lm_cached_analytic.npz",
        traj_xy=traj_final,
        beta=beta_final,
        resolution=resolution,
        origin_xy=np.array(origin_xy),
    )
    print("Saved optimized trajectory to optimized_traj_lm_cached_analytic.npz")


if __name__ == "__main__":
    main()

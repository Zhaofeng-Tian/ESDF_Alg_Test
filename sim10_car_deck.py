import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.patches import Polygon

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query


# =========================================================
# Geometry
# =========================================================
@dataclass(frozen=True)
class RectPart:
    name: str
    cx: float
    cy: float
    hx: float
    hy: float


def rect_vertices_local(rect: RectPart):
    x0 = rect.cx - rect.hx
    x1 = rect.cx + rect.hx
    y0 = rect.cy - rect.hy
    y1 = rect.cy + rect.hy
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=float,
    )


def sample_closed_polyline(vertices, step):
    if step <= 0.0:
        raise ValueError("step must be > 0")

    v = np.asarray(vertices, dtype=float)
    if v.shape[0] < 3:
        raise ValueError("Need at least 3 vertices.")

    v2 = np.vstack([v, v[0]])
    seg = v2[1:] - v2[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-12
    p0 = v2[:-1][keep]
    p1 = v2[1:][keep]
    seg_len = seg_len[keep]

    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = s_cum[-1]
    n = max(8, int(np.floor(perimeter / step)))
    s_vals = np.linspace(0.0, perimeter, n, endpoint=False)

    out = np.zeros((n, 2), dtype=float)
    j = 0
    for i, s in enumerate(s_vals):
        while j + 1 < s_cum.size and s_cum[j + 1] <= s:
            j += 1
        t = (s - s_cum[j]) / max(seg_len[j], 1e-12)
        out[i] = (1.0 - t) * p0[j] + t * p1[j]
    return out


def sample_rect_contour_with_vertices(vertices, step):
    """
    Mandatory-vertex contour sampling:
    - include all original vertices exactly
    - then add optional points along edges
    """
    if step <= 0.0:
        raise ValueError("step must be > 0")

    v = np.asarray(vertices, dtype=float)
    if v.shape[0] < 3:
        raise ValueError("Need at least 3 vertices.")

    extra = []
    n = v.shape[0]
    for i in range(n):
        p0 = v[i]
        p1 = v[(i + 1) % n]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        k = 1
        while True:
            s = k * step
            if s >= length - 1e-12:
                break
            t = s / length
            extra.append((1.0 - t) * p0 + t * p1)
            k += 1

    if len(extra) > 0:
        extra = np.asarray(extra, dtype=float)
        points = np.vstack([v, extra])
        is_vertex = np.concatenate(
            [np.ones(v.shape[0], dtype=bool), np.zeros(extra.shape[0], dtype=bool)]
        )
    else:
        points = v.copy()
        is_vertex = np.ones(v.shape[0], dtype=bool)

    return points, is_vertex


def rotate_points(points, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=float)
    return points @ r.T


def transform_points(local_pts, center_xy, theta):
    return center_xy + rotate_points(local_pts, theta)


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
# Obstacle + inside gradient with body/deck vertices
# =========================================================
def obstacle_grad_vertices(
    traj_xy,
    beta,
    sdf,
    origin_xy,
    resolution,
    d_target,
    margin_inside,
    body_points_local,
    deck_points_local,
    body_is_vertex_local,
    deck_is_vertex_local,
    return_point_debug=False,
):
    n = traj_xy.shape[0]

    grad_edge_xy = np.zeros_like(traj_xy)
    grad_edge_beta = np.zeros(n)

    grad_inside_xy = np.zeros_like(traj_xy)
    grad_inside_beta = np.zeros(n)

    dmin_deck = np.full(n, np.inf, dtype=float)
    dmin_all = np.full(n, np.inf, dtype=float)

    edge_cost = 0.0
    inside_cost = 0.0

    point_pos = []
    point_grad_edge_xy = []
    point_grad_inside_xy = []
    point_waypoint_idx = []
    point_is_deck = []
    point_is_vertex = []

    for i in range(n):
        th = beta[i]
        c = np.cos(th)
        s = np.sin(th)
        center = traj_xy[i]

        # Inside term on body points only.
        for j, (qx, qy) in enumerate(body_points_local):
            px = center[0] + c * qx - s * qy
            py = center[1] + s * qx + c * qy
            d, g = sdf_query(
                sdf,
                px,
                py,
                origin_xy=origin_xy,
                resolution=resolution,
            )

            dmin_all[i] = min(dmin_all[i], d)

            inside_vec_xy = np.zeros(2, dtype=float)
            if d < margin_inside:
                r_in = d - margin_inside
                inside_cost += 0.5 * r_in * r_in

                dpx_dtheta = -s * qx - c * qy
                dpy_dtheta = c * qx - s * qy
                inside_vec_xy = r_in * g
                grad_inside_xy[i] += inside_vec_xy
                grad_inside_beta[i] += r_in * (g[0] * dpx_dtheta + g[1] * dpy_dtheta)

            if return_point_debug:
                point_pos.append([px, py])
                point_grad_edge_xy.append([0.0, 0.0])
                point_grad_inside_xy.append(inside_vec_xy)
                point_waypoint_idx.append(i)
                point_is_deck.append(False)
                point_is_vertex.append(bool(body_is_vertex_local[j]))

        # Deck points: edge term + inside term.
        for j, (qx, qy) in enumerate(deck_points_local):
            px = center[0] + c * qx - s * qy
            py = center[1] + s * qx + c * qy
            d, g = sdf_query(
                sdf,
                px,
                py,
                origin_xy=origin_xy,
                resolution=resolution,
            )

            dmin_deck[i] = min(dmin_deck[i], d)
            dmin_all[i] = min(dmin_all[i], d)

            dpx_dtheta = -s * qx - c * qy
            dpy_dtheta = c * qx - s * qy

            r_edge = d - d_target
            edge_vec_xy = r_edge * g
            edge_cost += 0.5 * r_edge * r_edge
            grad_edge_xy[i] += edge_vec_xy
            grad_edge_beta[i] += r_edge * (g[0] * dpx_dtheta + g[1] * dpy_dtheta)

            inside_vec_xy = np.zeros(2, dtype=float)
            if d < margin_inside:
                r_in = d - margin_inside
                inside_cost += 0.5 * r_in * r_in
                inside_vec_xy = r_in * g
                grad_inside_xy[i] += inside_vec_xy
                grad_inside_beta[i] += r_in * (g[0] * dpx_dtheta + g[1] * dpy_dtheta)

            if return_point_debug:
                point_pos.append([px, py])
                point_grad_edge_xy.append(edge_vec_xy)
                point_grad_inside_xy.append(inside_vec_xy)
                point_waypoint_idx.append(i)
                point_is_deck.append(True)
                point_is_vertex.append(bool(deck_is_vertex_local[j]))

    if return_point_debug:
        point_pos = np.asarray(point_pos, dtype=float)
        point_grad_edge_xy = np.asarray(point_grad_edge_xy, dtype=float)
        point_grad_inside_xy = np.asarray(point_grad_inside_xy, dtype=float)
        point_waypoint_idx = np.asarray(point_waypoint_idx, dtype=int)
        point_is_deck = np.asarray(point_is_deck, dtype=bool)
        point_is_vertex = np.asarray(point_is_vertex, dtype=bool)
    else:
        point_pos = None
        point_grad_edge_xy = None
        point_grad_inside_xy = None
        point_waypoint_idx = None
        point_is_deck = None
        point_is_vertex = None

    return (
        dmin_deck,
        dmin_all,
        edge_cost,
        inside_cost,
        grad_edge_xy,
        grad_edge_beta,
        grad_inside_xy,
        grad_inside_beta,
        point_pos,
        point_grad_edge_xy,
        point_grad_inside_xy,
        point_waypoint_idx,
        point_is_deck,
        point_is_vertex,
    )


# =========================================================
# Smoothness
# =========================================================
def smoothness_grad(traj_xy):
    grad = np.zeros_like(traj_xy)
    grad[1:-1] = 2 * traj_xy[1:-1] - traj_xy[:-2] - traj_xy[2:]
    return grad


def smoothness_cost(traj_xy):
    diffs = traj_xy[1:] - traj_xy[:-1]
    return np.sum(np.linalg.norm(diffs, axis=1) ** 2)


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


def draw_gradient_quiver(
    ax,
    sdf,
    obstacle_mask,
    contour_px,
    traj_xy,
    beta,
    body_vertices_local,
    deck_vertices_local,
    point_pos_xy,
    point_grad_xy,
    point_is_deck,
    point_is_vertex,
    resolution,
    iter_idx,
):
    ax.clear()
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
    ax.plot(cpts[:, 0], cpts[:, 1], "w--", linewidth=1.2, label="Obstacle contour")
    ax.plot(
        traj_xy[:, 0] / resolution,
        traj_xy[:, 1] / resolution,
        color="black",
        linewidth=1.2,
        alpha=0.75,
        label="Current center path",
    )

    # Draw body/deck rectangles at every waypoint.
    for i in range(len(traj_xy)):
        center = traj_xy[i]
        th = beta[i]
        body_world = transform_points(body_vertices_local, center, th) / resolution
        deck_world = transform_points(deck_vertices_local, center, th) / resolution
        ax.add_patch(
            Polygon(
                body_world,
                closed=True,
                fill=False,
                edgecolor="#1f77b4",
                linewidth=0.6,
                alpha=0.35,
            )
        )
        ax.add_patch(
            Polygon(
                deck_world,
                closed=True,
                fill=False,
                edgecolor="#2ca02c",
                linewidth=0.6,
                alpha=0.35,
            )
        )

    pos_px = point_pos_xy / resolution
    grad_px = point_grad_xy / resolution
    body_mask = ~point_is_deck
    deck_mask = point_is_deck
    body_vertex_mask = body_mask & point_is_vertex
    deck_vertex_mask = deck_mask & point_is_vertex
    body_edge_mask = body_mask & (~point_is_vertex)
    deck_edge_mask = deck_mask & (~point_is_vertex)
    ax.quiver(
        pos_px[body_mask, 0],
        pos_px[body_mask, 1],
        grad_px[body_mask, 0],
        grad_px[body_mask, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#1f77b4",
        width=0.0018,
        alpha=0.8,
        label="Body point grad",
    )
    ax.quiver(
        pos_px[deck_mask, 0],
        pos_px[deck_mask, 1],
        grad_px[deck_mask, 0],
        grad_px[deck_mask, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#2ca02c",
        width=0.0018,
        alpha=0.9,
        label="Deck point grad",
    )
    ax.scatter(
        pos_px[body_edge_mask, 0],
        pos_px[body_edge_mask, 1],
        s=5,
        c="#1f77b4",
        alpha=0.35,
        label="Body edge samples",
    )
    ax.scatter(
        pos_px[deck_edge_mask, 0],
        pos_px[deck_edge_mask, 1],
        s=5,
        c="#2ca02c",
        alpha=0.35,
        label="Deck edge samples",
    )
    ax.scatter(
        pos_px[body_vertex_mask, 0],
        pos_px[body_vertex_mask, 1],
        s=26,
        marker="o",
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=1.1,
        alpha=0.95,
        label="Body vertices",
    )
    ax.scatter(
        pos_px[deck_vertex_mask, 0],
        pos_px[deck_vertex_mask, 1],
        s=26,
        marker="s",
        facecolors="none",
        edgecolors="#2ca02c",
        linewidths=1.1,
        alpha=0.95,
        label="Deck vertices",
    )
    ax.set_title(f"Gradient Arrows at Source Points | Iter {iter_idx}")
    ax.axis("off")
    ax.legend(loc="best")


# =========================================================
# MAIN
# =========================================================
def main():
    resolution = 0.05
    d_target = 0.06
    margin_inside = 0.0

    weights = {
        "edge": 0.0,
        "inside": 2.0,
        "smooth": 0.0,
        "h": 0.00,
    }

    sample_step_m = 0.05
    iters = 301
    history_stride = 10
    alpha = 0.05
    grad_plot_every = 10

    # Car center is body center in local frame.
    body_rect = RectPart(name="body", cx=0.00, cy=0.00, hx=0.42, hy=0.23)
    deck_rect = RectPart(name="deck", cx=0.44, cy=0.00, hx=0.22, hy=0.36)
    body_vertices_local = rect_vertices_local(body_rect)
    deck_vertices_local = rect_vertices_local(deck_rect)
    contour_step_vehicle = 0.20
    body_points_local, body_is_vertex_local = sample_rect_contour_with_vertices(
        body_vertices_local, contour_step_vehicle
    )
    deck_points_local, deck_is_vertex_local = sample_rect_contour_with_vertices(
        deck_vertices_local, contour_step_vehicle
    )

    # Acceptance checks (mandatory vertices).
    body_vertex_count = int(np.sum(body_is_vertex_local))
    deck_vertex_count = int(np.sum(deck_is_vertex_local))
    total_points = int(len(body_points_local) + len(deck_points_local))
    print(f"body_vertex_count={body_vertex_count}")
    print(f"deck_vertex_count={deck_vertex_count}")
    print(f"total_points={total_points}")
    if body_vertex_count != 4 or deck_vertex_count != 4:
        raise RuntimeError("Acceptance failed: body_vertex_count == 4 and deck_vertex_count == 4")
    tol = 1e-9
    body_vertices_present = all(
        np.any(np.linalg.norm(body_points_local - v, axis=1) <= tol) for v in body_vertices_local
    )
    deck_vertices_present = all(
        np.any(np.linalg.norm(deck_points_local - v, axis=1) <= tol) for v in deck_vertices_local
    )
    if not (body_vertices_present and deck_vertices_present):
        raise RuntimeError("Acceptance failed: not all rectangle vertices are in sampled points")

    obstacle_mask = load_occupancy_from_png(
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
    sample_step_px = sample_step_m / resolution
    traj_px = resample_closed_contour_uniform_px(contour_px, sample_step_px)
    traj_px = np.vstack([traj_px, traj_px[0]])  # closed loop seam

    traj_xy = traj_px * resolution
    beta = init_headings_from_path(traj_xy)
    origin_xy = (0.0, 0.0)

    print(f"Contour points: {len(contour_px)}")
    print(f"Sampling step: {sample_step_px:.2f} px ({sample_step_m:.3f} m)")
    print(f"Trajectory samples (closed): {len(traj_xy)}")
    print(f"Body contour sample points: {len(body_points_local)}")
    print(f"Deck contour sample points: {len(deck_points_local)}")

    traj_initial = traj_xy.copy()
    traj_history = [traj_xy.copy()]
    fig_grad, ax_grad = plt.subplots(figsize=(8, 8))

    for k in range(iters):
        need_point_debug = (k % grad_plot_every) == 0
        (
            dmin_deck,
            dmin_all,
            edge_cost,
            inside_cost,
            grad_edge_xy,
            grad_edge_beta,
            grad_inside_xy,
            grad_inside_beta,
            point_pos,
            point_grad_edge_xy,
            point_grad_inside_xy,
            point_waypoint_idx,
            point_is_deck,
            point_is_vertex,
        ) = obstacle_grad_vertices(
            traj_xy,
            beta,
            sdf,
            origin_xy,
            resolution,
            d_target,
            margin_inside,
            body_points_local,
            deck_points_local,
            body_is_vertex_local,
            deck_is_vertex_local,
            return_point_debug=need_point_debug,
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
            + weights["h"] * 100.0 * grad_h_beta
        )

        if need_point_debug and point_pos is not None:
            point_grad_xy = (
                weights["edge"] * point_grad_edge_xy
                + weights["inside"] * point_grad_inside_xy
            )
            seam_mask = (point_waypoint_idx == 0) | (point_waypoint_idx == (len(traj_xy) - 1))
            point_grad_xy[seam_mask] = 0.0
            draw_gradient_quiver(
                ax_grad,
                sdf,
                obstacle_mask,
                contour_px,
                traj_xy,
                beta,
                body_vertices_local,
                deck_vertices_local,
                point_pos,
                point_grad_xy,
                point_is_deck,
                point_is_vertex,
                resolution,
                k,
            )
            plt.pause(0.001)

        # Keep loop seam fixed.
        grad_xy[0] = 0.0
        grad_xy[-1] = 0.0
        grad_beta[0] = 0.0
        grad_beta[-1] = 0.0

        traj_xy -= alpha * grad_xy
        beta = wrap_to_pi(beta - alpha * grad_beta)

        if k % 50 == 0:
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
                f"smooth={smooth_cost:.6f}, h={h_term_cost:.6f}, total={total_cost:.6f}, "
                f"dmin_deck={np.min(dmin_deck):.4f}, dmin_all={np.min(dmin_all):.4f}"
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
    ax.plot(cpts[:, 0], cpts[:, 1], "w--", linewidth=1.5, label="Obstacle contour")

    ax.plot(
        traj_initial[:, 0] / resolution,
        traj_initial[:, 1] / resolution,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Initial center path",
    )

    for traj in traj_history[1:-1]:
        ax.plot(
            traj[:, 0] / resolution,
            traj[:, 1] / resolution,
            color="black",
            alpha=0.25,
            linewidth=1.0,
        )

    traj_final = traj_history[-1]
    ax.plot(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="blue",
        linewidth=2.3,
        label="Final center path",
    )
    ax.scatter(
        traj_final[:, 0] / resolution,
        traj_final[:, 1] / resolution,
        color="red",
        s=16,
        alpha=0.9,
        zorder=6,
        label="Final waypoints",
    )

    # Draw body/deck rectangles along trajectory (instead of circles).
    draw_stride = max(1, len(traj_final) // 35)
    for i in range(0, len(traj_final), draw_stride):
        center = traj_final[i]
        th = beta[i]

        body_world = transform_points(body_vertices_local, center, th) / resolution
        deck_world = transform_points(deck_vertices_local, center, th) / resolution

        ax.add_patch(
            Polygon(
                body_world,
                closed=True,
                fill=False,
                edgecolor="cyan",
                linewidth=0.9,
                alpha=0.9,
            )
        )
        ax.add_patch(
            Polygon(
                deck_world,
                closed=True,
                fill=False,
                edgecolor="magenta",
                linewidth=0.9,
                alpha=0.9,
            )
        )

    ax.set_title("sim10: Deck-Edge + All-Vertex-Inside Gradient Planning")
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

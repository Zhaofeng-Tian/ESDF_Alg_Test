# esdf.py
from __future__ import annotations
import numpy as np
from heapq import heappush, heappop

_NEIGHBORS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
    (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
]

def _dist_transform_2d(source_mask: np.ndarray, resolution: float, max_dist: float) -> np.ndarray:
    """
    Multi-source Dijkstra distance transform (8-connected).
    source_mask True cells have distance 0.
    Returns dist (meters), capped at max_dist.
    """
    H, W = source_mask.shape
    INF = 1e18
    dist = np.full((H, W), INF, dtype=float)

    pq: list[tuple[float, int, int]] = []
    for r, c in np.argwhere(source_mask):
        dist[r, c] = 0.0
        heappush(pq, (0.0, r, c))

    while pq:
        d, r, c = heappop(pq)
        if d != dist[r, c]:
            continue
        if d > max_dist:
            break
        for dr, dc, step in _NEIGHBORS_8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                nd = d + step * resolution
                if nd < dist[rr, cc] and nd <= max_dist:
                    dist[rr, cc] = nd
                    heappush(pq, (nd, rr, cc))

    return np.minimum(dist, max_dist)

def signed_esdf_2d(obstacle_mask: np.ndarray, *, resolution: float, max_dist: float) -> np.ndarray:
    """
    Signed ESDF (SDF on a grid):
      outside obstacles: +dist_to_obstacle
      inside obstacles:  -dist_to_free
    """
    obstacle_mask = obstacle_mask.astype(bool)
    free_mask = ~obstacle_mask

    d_out = _dist_transform_2d(obstacle_mask, resolution, max_dist)
    d_in = _dist_transform_2d(free_mask, resolution, max_dist)

    sdf = d_out.copy()
    sdf[obstacle_mask] = -d_in[obstacle_mask]
    return sdf

def _bilinear_sample(field: np.ndarray, x: float, y: float, origin_xy: tuple[float, float], resolution: float) -> float:
    """
    field indexed as field[row, col] where row increases with +y in array coords.
    origin_xy is world coordinate of the grid's (0,0) corner (min x, min y).
    """
    ox, oy = origin_xy
    H, W = field.shape

    gx = (x - ox) / resolution - 0.5  # col coordinate (float)
    gy = (y - oy) / resolution - 0.5  # row coordinate (float)

    # Clamp to valid interpolation region
    gx = float(np.clip(gx, 0.0, W - 1.001))
    gy = float(np.clip(gy, 0.0, H - 1.001))

    x0 = int(np.floor(gx)); y0 = int(np.floor(gy))
    x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1)
    wx = gx - x0; wy = gy - y0

    f00 = field[y0, x0]; f10 = field[y0, x1]
    f01 = field[y1, x0]; f11 = field[y1, x1]
    return (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11

def sdf_query(
    sdf_grid: np.ndarray,
    x: float,
    y: float,
    *,
    origin_xy: tuple[float, float],
    resolution: float,
) -> tuple[float, np.ndarray]:
    """
    Continuous query on the signed grid SDF:
      d(x,y) via bilinear interpolation
      grad via central differences on the interpolated field
    Returns (d, grad_xy).
    """
    d = _bilinear_sample(sdf_grid, x, y, origin_xy, resolution)
    eps = resolution
    dpx = _bilinear_sample(sdf_grid, x + eps, y, origin_xy, resolution)
    dmx = _bilinear_sample(sdf_grid, x - eps, y, origin_xy, resolution)
    dpy = _bilinear_sample(sdf_grid, x, y + eps, origin_xy, resolution)
    dmy = _bilinear_sample(sdf_grid, x, y - eps, origin_xy, resolution)
    grad = np.array([(dpx - dmx) / (2 * eps), (dpy - dmy) / (2 * eps)], dtype=float)
    return d, grad

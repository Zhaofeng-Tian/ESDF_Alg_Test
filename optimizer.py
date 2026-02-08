# optimizer.py
from __future__ import annotations
import numpy as np
from esdf import sdf_query

def _build_A_inv_second_diff(n_free: int, reg: float = 1e-4) -> np.ndarray:
    """
    Smoothness metric A = D^T D with second differences, inverted for preconditioning.
    """
    m = n_free - 2
    if m <= 0:
        return np.eye(n_free)

    D = np.zeros((m, n_free), dtype=float)
    for i in range(m):
        D[i, i] = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    A = D.T @ D + reg * np.eye(n_free)
    return np.linalg.inv(A)

def chompish_optimize_xy(
    p0: np.ndarray,
    pN: np.ndarray,
    sdf_grid: np.ndarray,
    *,
    origin_xy: tuple[float, float],
    resolution: float,
    N: int = 100,
    iters: int = 200,
    alpha: float = 0.03,
    w_obs: float = 20.0,
    robot_radius_m: float = 0.25,
    safe_dist_m: float = 0.45,
    grad_clip: float = 5.0,
    keep_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    CHOMP-ish path optimization in XY:
      delta = sdf(center) - robot_radius
      U = 0.5*(delta-safe)^2 if delta <= safe else 0
      grad = (delta-safe) * ∇sdf
      update: p_free <- p_free - alpha * A^{-1} * grad_free
    """
    xs = np.linspace(float(p0[0]), float(pN[0]), N)
    ys = np.linspace(float(p0[1]), float(pN[1]), N)

    idx_free = np.arange(1, N-1)
    A_inv = _build_A_inv_second_diff(len(idx_free))

    history: list[tuple[np.ndarray, np.ndarray]] = []
    if keep_history:
        history.append((xs.copy(), ys.copy()))

    for _k in range(iters):
        gx = np.zeros_like(xs)
        gy = np.zeros_like(ys)

        for i in idx_free:
            d, g = sdf_query(
                sdf_grid, float(xs[i]), float(ys[i]),
                origin_xy=origin_xy, resolution=resolution
            )
            delta = d - robot_radius_m
            if delta <= safe_dist_m:
                scale = (delta - safe_dist_m)  # dU/ddelta
                grad_xy = scale * g
                grad_xy = np.clip(grad_xy, -grad_clip, grad_clip)
                gx[i] += w_obs * grad_xy[0]
                gy[i] += w_obs * grad_xy[1]

        xs[idx_free] += -alpha * (A_inv @ gx[idx_free])
        ys[idx_free] += -alpha * (A_inv @ gy[idx_free])

        # endpoints fixed
        xs[0], ys[0] = p0
        xs[-1], ys[-1] = pN

        if keep_history:
            history.append((xs.copy(), ys.copy()))

    return xs, ys, history

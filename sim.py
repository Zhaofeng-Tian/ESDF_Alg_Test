# simulate.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_result(
    sdf_grid: np.ndarray,
    origin_xy: tuple[float, float],
    resolution: float,
    obstacle_outline_xy: np.ndarray | None,
    p0: np.ndarray,
    pN: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]] | None = None,
    show_checkpoints: bool = True,
):
    H, W = sdf_grid.shape
    extent = [
        origin_xy[0], origin_xy[0] + W * resolution,
        origin_xy[1] + H * resolution, origin_xy[1]
    ]

    plt.figure(figsize=(10, 6))
    plt.imshow(sdf_grid, origin="upper", extent=extent)
    plt.colorbar(label="Signed distance (m) (negative inside obstacle)")
    if obstacle_outline_xy is not None:
        poly = obstacle_outline_xy
        plt.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], linewidth=2)

    if history is not None and show_checkpoints:
        picks = [0, 5, 15, 30, 60, len(history)-1]
        picks = [k for k in picks if 0 <= k < len(history)]
        for k in picks:
            hx, hy = history[k]
            plt.plot(hx, hy, linewidth=1, label=f"iter {k}")

    plt.plot(xs, ys, linewidth=3, label="final")
    plt.scatter([p0[0], pN[0]], [p0[1], pN[1]], s=60)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

def animate_history(
    sdf_grid: np.ndarray,
    origin_xy: tuple[float, float],
    resolution: float,
    obstacle_outline_xy: np.ndarray | None,
    p0: np.ndarray,
    pN: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]],
    interval_ms: int = 40,
):
    H, W = sdf_grid.shape
    extent = [
        origin_xy[0], origin_xy[0] + W * resolution,
        origin_xy[1] + H * resolution, origin_xy[1]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(sdf_grid, origin="upper", extent=extent)
    if obstacle_outline_xy is not None:
        poly = obstacle_outline_xy
        ax.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], linewidth=2)

    ax.scatter([p0[0], pN[0]], [p0[1], pN[1]], s=60)
    (line,) = ax.plot(history[0][0], history[0][1], linewidth=3)
    title = ax.set_title("iter 0")
    ax.axis("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def update(frame: int):
        hx, hy = history[frame]
        line.set_data(hx, hy)
        title.set_text(f"iter {frame}")
        return line, title

    FuncAnimation(fig, update, frames=len(history), interval=interval_ms, blit=False)
    plt.show()

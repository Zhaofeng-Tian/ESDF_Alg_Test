import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass(frozen=True)
class RectPart:
    name: str
    cx: float
    cy: float
    hx: float
    hy: float


def rect_bounds(rect: RectPart):
    x0 = rect.cx - rect.hx
    x1 = rect.cx + rect.hx
    y0 = rect.cy - rect.hy
    y1 = rect.cy + rect.hy
    return x0, x1, y0, y1


def rect_vertices(rect: RectPart):
    x0, x1, y0, y1 = rect_bounds(rect)
    # Counter-clockwise from bottom-left
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=float,
    )


def closed(poly):
    return np.vstack([poly, poly[0]])


def sample_closed_polyline(vertices, step):
    if step <= 0.0:
        raise ValueError("step must be > 0")

    v = np.asarray(vertices, dtype=float)
    if v.shape[0] < 3:
        raise ValueError("Need at least 3 vertices.")

    v2 = closed(v)
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


def main():
    # Local robot frame: +x forward, +y left
    body = RectPart(name="body", cx=0.00, cy=0.00, hx=0.42, hy=0.23)
    deck = RectPart(name="deck", cx=0.44, cy=0.00, hx=0.22, hy=0.36)

    body_real_v = rect_vertices(body)
    deck_real_v = rect_vertices(deck)

    # Sample points along contour segments (not only vertices).
    contour_step = 0.06
    body_pts = sample_closed_polyline(body_real_v, contour_step)
    deck_pts = sample_closed_polyline(deck_real_v, contour_step)
    grad_pts = np.vstack([body_pts, deck_pts])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal", adjustable="box")

    # Contour segments (original shape only)
    ax.plot(*closed(body_real_v).T, "k-", lw=2.0, label="Body (real)")
    ax.plot(*closed(deck_real_v).T, "k-", lw=2.0, alpha=0.65, label="Deck (real)")

    # Gradient points sampled on contour segments
    ax.scatter(
        body_pts[:, 0],
        body_pts[:, 1],
        s=24,
        c="#1f77b4",
        label=f"Body contour samples ({len(body_pts)})",
        zorder=5,
    )
    ax.scatter(
        deck_pts[:, 0],
        deck_pts[:, 1],
        s=24,
        c="#2ca02c",
        label=f"Deck contour samples ({len(deck_pts)})",
        zorder=5,
    )

    ax.set_title("T-shape Contour Sampling (No Shrink)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from preprocess import load_occupancy_from_png
from mower_model import MowerFrontModel


def main():
    parser = argparse.ArgumentParser(description='Animate mower circle primitives along saved trajectory.')
    parser.add_argument('--traj', default='optimized_traj_lm_cached_analytic.npz', help='Saved trajectory npz file')
    parser.add_argument('--map', default='l_shape_obstacle_45deg.png', help='Occupancy map png used in optimization')
    parser.add_argument('--obstacle-is-dark', action='store_true', default=True, help='Treat dark pixels as obstacles')
    parser.add_argument('--circle-radius', type=float, default=0.05, help='Primitive circle radius in meters')
    parser.add_argument('--stride', type=int, default=1, help='Use every Nth trajectory sample for animation')
    parser.add_argument('--interval-ms', type=int, default=80, help='Delay between frames in milliseconds')
    parser.add_argument('--tail', type=int, default=25, help='Number of previous samples to show as a tail')
    parser.add_argument('--save-gif', default='', help='Optional output gif path')
    args = parser.parse_args()

    data = np.load(args.traj)
    traj_xy = data['traj_xy']
    beta = data['beta']
    resolution = float(data['resolution'])

    if args.stride > 1:
        traj_xy = traj_xy[::args.stride]
        beta = beta[::args.stride]

    obstacle_mask = load_occupancy_from_png(
        args.map,
        obstacle_is_dark=args.obstacle_is_dark,
        thresh=200,
    )

    robot = MowerFrontModel(circle_radius=args.circle_radius)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(obstacle_mask.astype(float), cmap='gray_r', origin='lower')

    path_line, = ax.plot([], [], 'b-', linewidth=2.0, alpha=0.85, label='optimized path')
    tail_line, = ax.plot([], [], 'r-', linewidth=2.0, alpha=0.9, label='robot tail')
    heading_line, = ax.plot([], [], 'g-', linewidth=2.0, alpha=0.9, label='heading')

    primitive_patches = []
    primitive_colors = {
        'collision': 'cyan',
        'working': 'magenta',
    }

    state0 = (float(traj_xy[0, 0]), float(traj_xy[0, 1]), float(beta[0]))
    for pr in robot.world_centers_and_jacobians(state0):
        patch = Circle(
            (pr['center_w'][0] / resolution, pr['center_w'][1] / resolution),
            radius=pr['radius'] / resolution,
            fill=False,
            edgecolor=primitive_colors.get(pr['tag'], 'yellow'),
            linewidth=2.0,
            alpha=0.95,
        )
        ax.add_patch(patch)
        primitive_patches.append((patch, pr['tag']))

    ax.set_title('Circle primitives along optimized trajectory')
    ax.axis('off')
    ax.legend(loc='upper right')

    xs_px = traj_xy[:, 0] / resolution
    ys_px = traj_xy[:, 1] / resolution
    margin = max(10, int(np.ceil(0.5 / resolution)))
    ax.set_xlim(xs_px.min() - margin, xs_px.max() + margin)
    ax.set_ylim(ys_px.min() - margin, ys_px.max() + margin)

    def update(frame_idx):
        x = float(traj_xy[frame_idx, 0])
        y = float(traj_xy[frame_idx, 1])
        b = float(beta[frame_idx])

        path_line.set_data(xs_px[: frame_idx + 1], ys_px[: frame_idx + 1])

        tail_start = max(0, frame_idx - args.tail)
        tail_line.set_data(xs_px[tail_start : frame_idx + 1], ys_px[tail_start : frame_idx + 1])

        head_len_m = 0.20
        hx0 = x / resolution
        hy0 = y / resolution
        hx1 = (x + head_len_m * np.cos(b)) / resolution
        hy1 = (y + head_len_m * np.sin(b)) / resolution
        heading_line.set_data([hx0, hx1], [hy0, hy1])

        prims = robot.world_centers_and_jacobians((x, y, b))
        for (patch, _), pr in zip(primitive_patches, prims):
            patch.center = (pr['center_w'][0] / resolution, pr['center_w'][1] / resolution)
            patch.radius = pr['radius'] / resolution
            patch.set_edgecolor(primitive_colors.get(pr['tag'], 'yellow'))

        ax.set_title(f'Circle primitives along optimized trajectory | frame {frame_idx + 1}/{len(traj_xy)}')
        return [path_line, tail_line, heading_line] + [p for p, _ in primitive_patches]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(traj_xy),
        interval=args.interval_ms,
        blit=True,
        repeat=True,
    )

    if args.save_gif:
        fps = max(1, int(round(1000.0 / args.interval_ms)))
        anim.save(args.save_gif, writer=PillowWriter(fps=fps))
        print(f'Saved gif to {args.save_gif}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
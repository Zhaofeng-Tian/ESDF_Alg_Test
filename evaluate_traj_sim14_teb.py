import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from preprocess import load_occupancy_from_png
from esdf import signed_esdf_2d, sdf_query
from mower_model import MowerFrontModel


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def safe_unit(v):
    l = float(np.linalg.norm(v))
    if l <= 1e-12:
        return np.zeros_like(v), 1e-12
    return v / l, l


def compute_pose_primitive_clearances(traj_xy, beta, robot, sdf, origin_xy, resolution):
    all_entries = []
    min_clearance = np.inf
    max_penetration = 0.0
    collision_count = 0

    for i in range(traj_xy.shape[0]):
        state = (float(traj_xy[i, 0]), float(traj_xy[i, 1]), float(beta[i]))
        prims = robot.world_centers_and_jacobians(state)
        pose_entries = []
        for pr in prims:
            c = pr["center_w"]
            d, g = sdf_query(
                sdf,
                float(c[0]),
                float(c[1]),
                origin_xy=origin_xy,
                resolution=resolution,
            )
            d_eff = float(d) - float(pr["radius"])
            pose_entries.append(
                {
                    "tag": pr["tag"],
                    "center_w": np.asarray(c, dtype=float),
                    "radius": float(pr["radius"]),
                    "d": float(d),
                    "g": np.asarray(g, dtype=float),
                    "d_eff": d_eff,
                }
            )
            min_clearance = min(min_clearance, d_eff)
            if d_eff < 0.0:
                collision_count += 1
                max_penetration = max(max_penetration, -d_eff)
        all_entries.append(pose_entries)

    return all_entries, float(min_clearance), int(collision_count), float(max_penetration)


def compute_teb_quantities(
    traj_xy,
    beta,
    dt,
    teb_kappa,
):
    n = traj_xy.shape[0]
    m = n - 1

    seg_len = np.zeros(m, dtype=float)
    proj = np.zeros(m, dtype=float)
    gamma = np.zeros(m, dtype=float)
    h = np.zeros(m, dtype=float)
    dtheta = np.zeros(m, dtype=float)
    v = np.zeros(m, dtype=float)
    omega = np.zeros(m, dtype=float)
    curvature = np.zeros(m, dtype=float)
    radius = np.full(m, np.inf, dtype=float)

    for k in range(m):
        seg = traj_xy[k + 1] - traj_xy[k]
        _, l = safe_unit(seg)
        dx, dy = float(seg[0]), float(seg[1])
        seg_len[k] = l

        cb = float(np.cos(beta[k]))
        sb = float(np.sin(beta[k]))
        proj[k] = cb * dx + sb * dy

        kp = teb_kappa * proj[k]
        gamma[k] = kp / (1.0 + abs(kp))

        cbk = float(np.cos(beta[k]))
        sbk = float(np.sin(beta[k]))
        cbk1 = float(np.cos(beta[k + 1]))
        sbk1 = float(np.sin(beta[k + 1]))
        h[k] = (cbk + cbk1) * dy - (sbk + sbk1) * dx

        dtheta[k] = wrap_to_pi(beta[k + 1] - beta[k])
        dt_k = max(float(dt[k]), 1e-9)
        v[k] = l * gamma[k] / dt_k
        omega[k] = dtheta[k] / dt_k

        if l > 1e-12:
            curvature[k] = abs(dtheta[k]) / l
        else:
            curvature[k] = np.inf

        if abs(dtheta[k]) > 1e-12:
            radius[k] = l / abs(dtheta[k])

    if m >= 2:
        accel = np.zeros(m - 1, dtype=float)
        alpha = np.zeros(m - 1, dtype=float)
        for k in range(m - 1):
            denom = max(float(dt[k] + dt[k + 1]), 1e-9)
            accel[k] = 2.0 * (v[k + 1] - v[k]) / denom
            alpha[k] = 2.0 * (omega[k + 1] - omega[k]) / denom
    else:
        accel = np.zeros(0, dtype=float)
        alpha = np.zeros(0, dtype=float)

    return {
        "seg_len": seg_len,
        "proj": proj,
        "gamma": gamma,
        "h": h,
        "dtheta": dtheta,
        "v": v,
        "omega": omega,
        "accel": accel,
        "alpha": alpha,
        "curvature": curvature,
        "radius": radius,
    }


def violation_stats(values, limit, use_abs=True):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "limit": float(limit),
            "count": 0,
            "max_value": 0.0,
            "max_excess": 0.0,
            "mean_excess_active": 0.0,
        }

    eval_arr = np.abs(arr) if use_abs else arr
    excess = np.maximum(eval_arr - float(limit), 0.0)
    active = excess > 0.0
    return {
        "limit": float(limit),
        "count": int(np.count_nonzero(active)),
        "max_value": float(np.max(eval_arr)),
        "max_excess": float(np.max(excess)),
        "mean_excess_active": float(np.mean(excess[active])) if np.any(active) else 0.0,
    }


def turning_radius_stats(radius, min_turning_radius):
    radius = np.asarray(radius, dtype=float)
    finite = np.isfinite(radius)
    valid = radius[finite]
    if valid.size == 0:
        return {
            "limit": float(min_turning_radius),
            "count": 0,
            "min_value": float("inf"),
            "max_shortfall": 0.0,
            "mean_shortfall_active": 0.0,
        }

    if min_turning_radius <= 0.0:
        return {
            "limit": float(min_turning_radius),
            "count": 0,
            "min_value": float(np.min(valid)),
            "max_shortfall": 0.0,
            "mean_shortfall_active": 0.0,
        }

    shortfall = np.maximum(float(min_turning_radius) - valid, 0.0)
    active = shortfall > 0.0
    return {
        "limit": float(min_turning_radius),
        "count": int(np.count_nonzero(active)),
        "min_value": float(np.min(valid)),
        "max_shortfall": float(np.max(shortfall)),
        "mean_shortfall_active": float(np.mean(shortfall[active])) if np.any(active) else 0.0,
    }


def summarize_metrics(metrics, clearance, collision_count, max_penetration, d_safe):
    h_abs = np.abs(metrics["h"])
    return {
        "trajectory": {
            "num_poses": int(len(metrics["v"]) + 1),
            "num_segments": int(len(metrics["v"])),
            "path_length_m": float(np.sum(metrics["seg_len"])),
            "total_time_s": float(np.sum(dt_global)),
        },
        "collision": {
            "min_esdf_clearance_m": float(clearance),
            "collision_primitive_count": int(collision_count),
            "max_penetration_m": float(max_penetration),
            "d_safe_m": float(d_safe),
            "safety_margin_violations": int(np.count_nonzero(clearance_per_primitive_global < d_safe)),
        },
        "nonholonomic": {
            "max_abs_h": float(np.max(h_abs)) if h_abs.size else 0.0,
            "mean_abs_h": float(np.mean(h_abs)) if h_abs.size else 0.0,
            "rms_h": float(np.sqrt(np.mean(metrics["h"] ** 2))) if h_abs.size else 0.0,
        },
        "velocity": violation_stats(metrics["v"], args_global.v_max, use_abs=True),
        "omega": violation_stats(metrics["omega"], args_global.omega_max, use_abs=True),
        "accel": violation_stats(metrics["accel"], args_global.a_max, use_abs=True),
        "alpha": violation_stats(metrics["alpha"], args_global.alpha_max, use_abs=True),
        "curvature": {
            "max_value": float(np.max(metrics["curvature"])) if metrics["curvature"].size else 0.0,
            "mean_value": float(np.mean(metrics["curvature"])) if metrics["curvature"].size else 0.0,
        },
        "turning_radius": turning_radius_stats(metrics["radius"], args_global.min_turning_radius),
    }


def print_summary(summary):
    print("\n=== TRAJECTORY EVALUATION SUMMARY ===")
    print(
        f"poses={summary['trajectory']['num_poses']} | "
        f"segments={summary['trajectory']['num_segments']} | "
        f"path_length={summary['trajectory']['path_length_m']:.4f} m | "
        f"total_time={summary['trajectory']['total_time_s']:.4f} s"
    )
    print(
        f"clearance_min={summary['collision']['min_esdf_clearance_m']:.4f} m | "
        f"primitive_collisions={summary['collision']['collision_primitive_count']} | "
        f"max_penetration={summary['collision']['max_penetration_m']:.4f} m | "
        f"safe_margin_violations={summary['collision']['safety_margin_violations']}"
    )
    print(
        f"H-term: max|h|={summary['nonholonomic']['max_abs_h']:.6f} | "
        f"mean|h|={summary['nonholonomic']['mean_abs_h']:.6f} | "
        f"rms={summary['nonholonomic']['rms_h']:.6f}"
    )
    for key in ["velocity", "omega", "accel", "alpha"]:
        block = summary[key]
        print(
            f"{key}: limit={block['limit']:.4f} | count={block['count']} | "
            f"max_value={block['max_value']:.6f} | max_excess={block['max_excess']:.6f}"
        )
    tr = summary["turning_radius"]
    print(
        f"turning_radius: limit={tr['limit']:.4f} | count={tr['count']} | "
        f"min_value={tr['min_value']:.6f} | max_shortfall={tr['max_shortfall']:.6f}"
    )
    curv = summary["curvature"]
    print(f"curvature: max={curv['max_value']:.6f} 1/m | mean={curv['mean_value']:.6f} 1/m")


def make_plots(
    obstacle_mask,
    sdf,
    traj_xy,
    beta,
    primitive_cache,
    metrics,
    resolution,
    save_prefix,
    show_primitives_stride,
):
    sdf_vis = sdf.copy()
    sdf_vis[obstacle_mask] = -1.0

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    im = ax1.imshow(sdf_vis, cmap="RdYlGn", vmin=-1.0, vmax=max(1.0, float(np.nanmax(sdf_vis))), origin="lower")
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="ESDF [m]")
    ax1.plot(traj_xy[:, 0] / resolution, traj_xy[:, 1] / resolution, color="blue", linewidth=2.2, label="trajectory")
    ax1.scatter(traj_xy[:, 0] / resolution, traj_xy[:, 1] / resolution, s=10, color="red", alpha=0.9, zorder=5)
    arrow_scale = 0.05
    ax1.quiver(
        traj_xy[:, 0] / resolution,
        traj_xy[:, 1] / resolution,
        arrow_scale * np.cos(beta) / resolution,
        arrow_scale * np.sin(beta) / resolution,
        color="white",
        scale=5.0,
        scale_units="xy",
        width=0.003,
    )

    stride = max(1, int(show_primitives_stride))
    for i in range(0, len(primitive_cache), stride):
        for pr in primitive_cache[i]:
            ec = "cyan" if pr["tag"] == "collision" else "magenta"
            ls = "-" if pr["d_eff"] >= 0.0 else "--"
            ax1.add_patch(
                Circle(
                    (pr["center_w"][0] / resolution, pr["center_w"][1] / resolution),
                    radius=pr["radius"] / resolution,
                    fill=False,
                    edgecolor=ec,
                    linewidth=0.7,
                    alpha=0.8,
                    linestyle=ls,
                )
            )

    ax1.set_title("Trajectory over ESDF")
    ax1.axis("off")
    ax1.legend(loc="best")
    fig1.tight_layout()

    fig2, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    series = [
        (metrics["h"], "h term", 0),
        (metrics["v"], "v [m/s]", args_global.v_max),
        (metrics["omega"], "omega [rad/s]", args_global.omega_max),
        (metrics["accel"], "a [m/s^2]", args_global.a_max),
        (metrics["alpha"], "alpha [rad/s^2]", args_global.alpha_max),
        (metrics["radius"], "turning radius [m]", args_global.min_turning_radius),
    ]
    for ax, (vals, title, limit) in zip(axes, series):
        vals = np.asarray(vals, dtype=float)
        ax.plot(vals, linewidth=1.8)
        if title == "turning radius [m]":
            finite = np.isfinite(vals)
            if np.any(~finite):
                ax.plot(np.where(~finite)[0], np.zeros(np.count_nonzero(~finite)), "rx", label="inf radius")
            if limit > 0.0:
                ax.axhline(limit, linestyle="--", linewidth=1.2, label="limit")
        else:
            if limit > 0.0:
                ax.axhline(limit, linestyle="--", linewidth=1.2, label="+limit")
                if title != "h term":
                    ax.axhline(-limit, linestyle="--", linewidth=1.2, label="-limit")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if limit > 0.0:
            ax.legend(loc="best")

    fig2.tight_layout()

    if save_prefix:
        p = Path(save_prefix)
        fig1.savefig(str(p.with_name(p.name + "_traj_esdf.png")), dpi=180, bbox_inches="tight")
        fig2.savefig(str(p.with_name(p.name + "_profiles.png")), dpi=180, bbox_inches="tight")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved TEB-style trajectory constraints and ESDF collision margins.")
    parser.add_argument("--traj", default="optimized_traj_lm_teb_style.npz", help="Saved trajectory npz")
    parser.add_argument("--map", default="l_shape_obstacle_45deg.png", help="Occupancy map png from sim14_TEB_LM2.py")
    parser.add_argument("--obstacle-is-dark", action="store_true", default=True, help="Treat dark pixels as obstacles")
    parser.add_argument("--resolution", type=float, default=None, help="Override map resolution [m/pixel]")
    parser.add_argument("--origin-x", type=float, default=None, help="Override origin x [m]")
    parser.add_argument("--origin-y", type=float, default=None, help="Override origin y [m]")
    parser.add_argument("--circle-radius", type=float, default=0.05, help="Robot primitive circle radius [m]")
    parser.add_argument("--d-safe", type=float, default=0.02, help="Safety distance [m] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--teb-kappa", type=float, default=100.0, help="Gamma smoothing scale used in the optimizer")
    parser.add_argument("--v-max", type=float, default=0.40, help="Velocity limit [m/s] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--omega-max", type=float, default=0.8, help="Angular velocity limit [rad/s] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--a-max", type=float, default=0.8, help="Acceleration limit [m/s^2] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--alpha-max", type=float, default=2.0, help="Angular acceleration limit [rad/s^2] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--min-turning-radius", type=float, default=1.0, help="Minimum steering radius [m] (matches sim14_TEB_LM2.py)")
    parser.add_argument("--save-prefix", default="eval_sim14", help="Optional prefix for saved summary/plots")
    parser.add_argument("--show-primitives-stride", type=int, default=8, help="Draw every Nth pose primitive on the ESDF plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args_global = args
    print("Using evaluation defaults aligned with sim14_TEB_LM2.py")
    print(f"map={args.map} | d_safe={args.d_safe} | v_max={args.v_max} | omega_max={args.omega_max} | a_max={args.a_max} | alpha_max={args.alpha_max} | min_turning_radius={args.min_turning_radius}")

    data = np.load(args.traj, allow_pickle=False)
    traj_xy = np.asarray(data["traj_xy"], dtype=float)
    beta = np.asarray(data["beta"], dtype=float)
    if "dt" not in data:
        raise KeyError("The trajectory file does not contain 'dt'. Re-run the TEB-style optimizer that saves dt.")
    dt = np.asarray(data["dt"], dtype=float)
    dt_global = dt

    resolution = float(data["resolution"]) if "resolution" in data and args.resolution is None else float(args.resolution)
    if "origin_xy" in data and args.origin_x is None and args.origin_y is None:
        origin_xy = tuple(np.asarray(data["origin_xy"], dtype=float).tolist())
    else:
        ox = 0.0 if args.origin_x is None else float(args.origin_x)
        oy = 0.0 if args.origin_y is None else float(args.origin_y)
        origin_xy = (ox, oy)

    obstacle_mask = load_occupancy_from_png(
        args.map,
        obstacle_is_dark=args.obstacle_is_dark,
        thresh=200,
    )
    sdf = signed_esdf_2d(obstacle_mask, resolution=resolution, max_dist=5.0)
    robot = MowerFrontModel(circle_radius=args.circle_radius)

    primitive_cache, min_clearance, collision_count, max_penetration = compute_pose_primitive_clearances(
        traj_xy, beta, robot, sdf, origin_xy, resolution
    )
    clearance_per_primitive = [pr["d_eff"] for pose in primitive_cache for pr in pose]
    clearance_per_primitive_global = np.asarray(clearance_per_primitive, dtype=float)

    metrics = compute_teb_quantities(traj_xy, beta, dt, args.teb_kappa)
    summary = summarize_metrics(metrics, min_clearance, collision_count, max_penetration, args.d_safe)
    print_summary(summary)

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        summary_path = prefix.with_name(prefix.name + "_summary.json")
        metrics_path = prefix.with_name(prefix.name + "_metrics.npz")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        np.savez(
            metrics_path,
            traj_xy=traj_xy,
            beta=beta,
            dt=dt,
            clearance_per_primitive=clearance_per_primitive_global,
            **{k: np.asarray(v) for k, v in metrics.items()},
        )
        print(f"Saved summary to {summary_path}")
        print(f"Saved metrics to {metrics_path}")

    make_plots(
        obstacle_mask,
        sdf,
        traj_xy,
        beta,
        primitive_cache,
        metrics,
        resolution,
        args.save_prefix,
        args.show_primitives_stride,
    )

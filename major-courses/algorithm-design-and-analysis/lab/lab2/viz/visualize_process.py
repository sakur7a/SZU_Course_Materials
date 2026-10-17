import argparse
import csv
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


@dataclass
class Point:
    idx: int
    x: float
    y: float


@dataclass
class TraceStep:
    step: int
    id_a: int
    id_b: int
    distance: float
    phase: str


@dataclass
class DivisionStep:
    depth: int
    left: int
    right: int
    mid_x: float
    strip_left: float
    strip_right: float
    best_dist: float


def load_points(path):
    points = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(Point(int(row["id"]), float(row["x"]), float(row["y"])))
    return points


def load_pair(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        row = rows[0]
        return int(row["id_a"]), int(row["id_b"]), float(row["distance"])


def load_trace(path):
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(
                TraceStep(
                    step=int(row["step"]),
                    id_a=int(row["id_a"]),
                    id_b=int(row["id_b"]),
                    distance=float(row["distance"]),
                    phase=row.get("phase", "trace"),
                )
            )
    steps.sort(key=lambda s: s.step)
    return steps


def load_division_steps(path):
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(
                DivisionStep(
                    depth=int(row["depth"]),
                    left=int(row["left"]),
                    right=int(row["right"]),
                    mid_x=float(row["mid_x"]),
                    strip_left=float(row["strip_left"]),
                    strip_right=float(row["strip_right"]),
                    best_dist=float(row["best_dist"]),
                )
            )
    return steps


def brute_force_steps(points, max_frames):
    best = float("inf")
    best_pair = None
    frames = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i].x - points[j].x
            dy = points[i].y - points[j].y
            d = math.hypot(dx, dy)
            if d < best:
                best = d
                best_pair = (points[i].idx, points[j].idx)
                frames.append((i, j, best_pair, best))
                if len(frames) >= max_frames:
                    return frames
    return frames


def make_steps(points, trace_steps, max_frames):
    if trace_steps:
        raw = trace_steps[: max_frames]
        normalized = []
        for s in raw:
            normalized.append(
                {
                    "id_a": s.id_a,
                    "id_b": s.id_b,
                    "distance": s.distance,
                    "phase": s.phase,
                }
            )
        return normalized, "trace"

    raw = brute_force_steps(points, max_frames)
    normalized = []
    for _, _, best_pair, best in raw:
        normalized.append(
            {
                "id_a": best_pair[0],
                "id_b": best_pair[1],
                "distance": best,
                "phase": "replay",
            }
        )
    return normalized, "replay"


def run_classic(points, pair, steps, point_map, args, mode_label):
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#fdfefe")

    xs = [p.x for p in points]
    ys = [p.y for p in points]

    ax.scatter(xs, ys, s=28, c="#1d4ed8", alpha=0.82, edgecolors="#dbeafe", linewidths=0.4)
    ax.set_title(f"Closest Pair Process Visualization ({mode_label})", fontsize=15, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25, color="#cbd5e1")

    line, = ax.plot([], [], color="#ef4444", linewidth=2.6, alpha=0.95, zorder=4)
    pulse_a = ax.scatter([], [], s=130, color="#fb7185", alpha=0.35, zorder=5)
    pulse_b = ax.scatter([], [], s=130, color="#fb7185", alpha=0.35, zorder=5)
    pin_a = ax.scatter([], [], s=48, color="#be123c", edgecolors="#fff1f2", linewidths=0.8, zorder=6)
    pin_b = ax.scatter([], [], s=48, color="#be123c", edgecolors="#fff1f2", linewidths=0.8, zorder=6)

    info_box = FancyBboxPatch((0.015, 0.89), 0.43, 0.105, boxstyle="round,pad=0.02,rounding_size=0.015", transform=ax.transAxes, linewidth=0.8, edgecolor="#cbd5e1", facecolor="#ffffff", alpha=0.9)
    ax.add_patch(info_box)
    text = ax.text(0.03, 0.965, "", transform=ax.transAxes, va="top", fontsize=10, color="#0f172a")

    def init():
        line.set_data([], [])
        pulse_a.set_offsets([[math.nan, math.nan]])
        pulse_b.set_offsets([[math.nan, math.nan]])
        pin_a.set_offsets([[math.nan, math.nan]])
        pin_b.set_offsets([[math.nan, math.nan]])
        text.set_text("Preparing animation...")
        return line, pulse_a, pulse_b, pin_a, pin_b, text

    def update(frame_idx):
        s = steps[frame_idx]
        best_pair = (s["id_a"], s["id_b"])
        best = s["distance"]
        phase = s["phase"]

        pa = point_map[best_pair[0]]
        pb = point_map[best_pair[1]]
        line.set_data([pa.x, pb.x], [pa.y, pb.y])
        pulse_scale = 120 + 70 * (1.0 + math.sin(frame_idx * 0.55))
        pulse_a.set_sizes([pulse_scale])
        pulse_b.set_sizes([pulse_scale])
        pulse_a.set_offsets([[pa.x, pa.y]])
        pulse_b.set_offsets([[pb.x, pb.y]])
        pin_a.set_offsets([[pa.x, pa.y]])
        pin_b.set_offsets([[pb.x, pb.y]])
        text.set_text(
            f"Frame: {frame_idx + 1}/{len(steps)}\n"
            f"Current best distance: {best:.6f}\n"
            f"Pair IDs: ({best_pair[0]}, {best_pair[1]})  phase={phase}"
        )
        return line, pulse_a, pulse_b, pin_a, pin_b, text

    ani = None
    if steps:
        ani = FuncAnimation(fig, update, frames=len(steps), init_func=init, interval=260, blit=True, repeat=False)
    else:
        text.set_text("No update frames generated.")

    if pair is not None:
        a, b, d = pair
        pa = point_map[a]
        pb = point_map[b]
        ax.plot([pa.x, pb.x], [pa.y, pb.y], color="#16a34a", linewidth=3.0, alpha=0.9, zorder=7)
        ax.scatter([pa.x, pb.x], [pa.y, pb.y], s=64, color="#16a34a", edgecolors="#dcfce7", linewidths=1.0, zorder=8)
        ax.text(0.03, 0.865, f"Final closest distance: {d:.6f}", transform=ax.transAxes, va="top", fontsize=10, color="#166534")

    xpad = (max(xs) - min(xs)) * 0.08 if len(xs) > 1 else 1.0
    ypad = (max(ys) - min(ys)) * 0.08 if len(ys) > 1 else 1.0
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    return fig, ani


def run_dashboard(points, pair, steps, point_map, args, mode_label, division_steps=None):
    if division_steps is None:
        division_steps = []

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    fig = plt.figure(figsize=(14.4, 8.2))
    fig.patch.set_facecolor("#f3f6fb")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.16, hspace=0.32)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[1, 1])

    ax_main.set_facecolor("#ffffff")
    ax_curve.set_facecolor("#ffffff")
    ax_zoom.set_facecolor("#ffffff")

    ax_main.scatter(xs, ys, s=24, c="#2563eb", alpha=0.75, edgecolors="#dbeafe", linewidths=0.35)
    ax_main.set_title(f"Global View ({mode_label})", fontsize=13, fontweight="bold")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.grid(alpha=0.25, color="#d7deea")

    xpad = (max(xs) - min(xs)) * 0.08 if len(xs) > 1 else 1.0
    ypad = (max(ys) - min(ys)) * 0.08 if len(ys) > 1 else 1.0
    ax_main.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax_main.set_ylim(min(ys) - ypad, max(ys) + ypad)

    moving_line, = ax_main.plot([], [], color="#ef4444", linewidth=2.4, zorder=5)
    moving_pts = ax_main.scatter([], [], s=72, color="#ef4444", edgecolors="#fee2e2", linewidths=1.0, zorder=6)

    # 分治几何元素：中线、strip、左右子问题边界
    divcon_midline = ax_main.axvline(x=0, color="#f59e0b", linewidth=1.7, linestyle="--", alpha=0.82, zorder=3)
    divcon_strip = Rectangle((0, 0), 0, 0, color="#fbbf24", alpha=0.18, zorder=1)
    ax_main.add_patch(divcon_strip)
    subproblem_span = Rectangle((0, 0), 0, 0, color="#60a5fa", alpha=0.08, zorder=0)
    ax_main.add_patch(subproblem_span)
    left_boundary_line = ax_main.axvline(x=0, color="#2563eb", linewidth=1.2, linestyle=":", alpha=0.95, zorder=2)
    right_boundary_line = ax_main.axvline(x=0, color="#2563eb", linewidth=1.2, linestyle=":", alpha=0.95, zorder=2)
    geom_text = ax_main.text(
        0.98,
        0.97,
        "",
        transform=ax_main.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
    )

    points_sorted_x = sorted(points, key=lambda p: p.x)

    final_line = None
    final_dist_text = None
    if pair is not None:
        a, b, d = pair
        pa = point_map[a]
        pb = point_map[b]
        final_line, = ax_main.plot([pa.x, pb.x], [pa.y, pb.y], color="#16a34a", linewidth=2.8, alpha=0.85, zorder=4)
        ax_main.scatter([pa.x, pb.x], [pa.y, pb.y], s=66, color="#16a34a", edgecolors="#dcfce7", linewidths=0.9, zorder=7)
        final_dist_text = ax_main.text(
            0.02,
            0.97,
            f"Final: {d:.4f}",
            transform=ax_main.transAxes,
            va="top",
            color="#166534",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=1),
        )

    dists = [s["distance"] for s in steps]
    indices = list(range(1, len(steps) + 1))
    ax_curve.set_title("Best Distance vs Update Step", fontsize=12, fontweight="bold")
    ax_curve.set_xlabel("update step", labelpad=8)
    ax_curve.set_ylabel("best distance")
    ax_curve.grid(alpha=0.25, color="#d7deea")
    if dists:
        ax_curve.plot(indices, dists, color="#0f766e", linewidth=2.0, drawstyle="steps-post", alpha=0.85)
        marker_curve, = ax_curve.plot([], [], marker="o", color="#ef4444", markersize=7)
        ax_curve.set_xlim(1, max(indices))
        low = min(dists)
        high = max(dists)
        pad = max((high - low) * 0.12, 1e-9)
        ax_curve.set_ylim(low - pad, high + pad)
    else:
        marker_curve, = ax_curve.plot([], [], marker="o", color="#ef4444", markersize=7)
        ax_curve.text(0.06, 0.88, "No steps", transform=ax_curve.transAxes, color="#64748b")

    ax_zoom.set_title("Local Zoom Around Current Pair", fontsize=12, fontweight="bold", pad=8)
    ax_zoom.set_xlabel("x")
    ax_zoom.set_ylabel("y")
    ax_zoom.grid(alpha=0.25, color="#d7deea")

    zoom_scatter = ax_zoom.scatter([], [], s=28, color="#64748b", alpha=0.72)
    zoom_line, = ax_zoom.plot([], [], color="#ef4444", linewidth=2.4)
    zoom_pts = ax_zoom.scatter([], [], s=74, color="#ef4444", edgecolors="#fee2e2", linewidths=1.0)
    status = ax_zoom.text(0.02, 0.97, "", transform=ax_zoom.transAxes, va="top", fontsize=7.8, color="#0f172a", bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.92))

    fig.suptitle("Closest Pair Process Dashboard", fontsize=18, fontweight="bold", y=0.985)

    def update(frame_idx):
        s = steps[frame_idx]
        a = s["id_a"]
        b = s["id_b"]
        d = s["distance"]
        phase = s["phase"]
        pa = point_map[a]
        pb = point_map[b]

        moving_line.set_data([pa.x, pb.x], [pa.y, pb.y])
        moving_pts.set_offsets([[pa.x, pa.y], [pb.x, pb.y]])

        # 更新分治几何信息（按动画帧映射到分治步骤）
        if division_steps and len(division_steps) > 0:
            if len(steps) <= 1:
                div_idx = len(division_steps) - 1
            else:
                div_idx = int((frame_idx / (len(steps) - 1)) * (len(division_steps) - 1))
            cur_div = division_steps[div_idx]

            mid = cur_div.mid_x
            strip_left = cur_div.strip_left
            strip_right = cur_div.strip_right
            strip_width = max(0.0, strip_right - strip_left)

            # 左右子问题边界：由 x 排序后的区间索引映射到坐标
            n_sorted = len(points_sorted_x)
            left_idx = min(max(cur_div.left, 0), n_sorted - 1)
            right_exclusive = min(max(cur_div.right, 1), n_sorted)
            right_idx = max(left_idx, right_exclusive - 1)
            left_x = points_sorted_x[left_idx].x
            right_x = points_sorted_x[right_idx].x

            # 中线 L
            divcon_midline.set_xdata([mid, mid])

            # strip 带状区
            divcon_strip.set_xy((strip_left, min(ys) - ypad))
            divcon_strip.set_width(strip_width)
            divcon_strip.set_height(max(ys) - min(ys) + 2 * ypad)

            # 左右子问题边界（区间）
            left_boundary_line.set_xdata([left_x, left_x])
            right_boundary_line.set_xdata([right_x, right_x])
            subproblem_span.set_xy((left_x, min(ys) - ypad))
            subproblem_span.set_width(max(0.0, right_x - left_x))
            subproblem_span.set_height(max(ys) - min(ys) + 2 * ypad)

            geom_text.set_text(
                f"depth={cur_div.depth}  range=[{cur_div.left},{cur_div.right})\n"
                f"L={mid:.1f}  strip=[{strip_left:.1f}, {strip_right:.1f}]"
            )
        else:
            divcon_midline.set_xdata([math.nan, math.nan])
            left_boundary_line.set_xdata([math.nan, math.nan])
            right_boundary_line.set_xdata([math.nan, math.nan])
            divcon_strip.set_width(0)
            subproblem_span.set_width(0)
            geom_text.set_text("no division steps")

        if dists:
            marker_curve.set_data([frame_idx + 1], [d])

        cx = (pa.x + pb.x) / 2.0
        cy = (pa.y + pb.y) / 2.0
        radius = max(d * 5.5, 30.0)
        ax_zoom.set_xlim(cx - radius, cx + radius)
        ax_zoom.set_ylim(cy - radius, cy + radius)

        local_x = []
        local_y = []
        for p in points:
            if abs(p.x - cx) <= radius and abs(p.y - cy) <= radius:
                local_x.append(p.x)
                local_y.append(p.y)
        if local_x:
            zoom_scatter.set_offsets([[xx, yy] for xx, yy in zip(local_x, local_y)])
        else:
            zoom_scatter.set_offsets([[math.nan, math.nan]])

        zoom_line.set_data([pa.x, pb.x], [pa.y, pb.y])
        zoom_pts.set_offsets([[pa.x, pa.y], [pb.x, pb.y]])
        status.set_text(
            f"frame={frame_idx + 1}/{len(steps)}  phase={phase}\n"
            f"pair=({a}, {b})  dist={d:.6f}"
        )

        artists = [
            moving_line,
            moving_pts,
            marker_curve,
            zoom_scatter,
            zoom_line,
            zoom_pts,
            status,
        ]
        if final_line is not None:
            artists.append(final_line)
        return tuple(artists)

    ani = None
    if steps:
        ani = FuncAnimation(fig, update, frames=len(steps), interval=220, blit=False, repeat=False)
        # 让静态导出的 PNG 也呈现有效过程帧，而不是初始空状态。
        preview_idx = len(steps) - 1 if len(steps) < 4 else int(0.72 * (len(steps) - 1))
        update(preview_idx)
    else:
        status.set_text("No update frames generated.")

    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--trace", default="")
    parser.add_argument("--steps", default="")
    parser.add_argument("--view", choices=["dashboard", "classic"], default="dashboard")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--save", default="")
    parser.add_argument("--save-gif", default="")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    points = load_points(args.points)
    pair = load_pair(args.pair)
    point_map = {p.idx: p for p in points}

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#fdfefe")

    xs = [p.x for p in points]
    ys = [p.y for p in points]

    ax.scatter(xs, ys, s=28, c="#1d4ed8", alpha=0.82, edgecolors="#dbeafe", linewidths=0.4)
    ax.set_title("Closest Pair Process Visualization", fontsize=15, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25, color="#cbd5e1")

    line, = ax.plot([], [], color="#ef4444", linewidth=2.6, alpha=0.95, zorder=4)
    pulse_a = ax.scatter([], [], s=130, color="#fb7185", alpha=0.35, zorder=5)
    pulse_b = ax.scatter([], [], s=130, color="#fb7185", alpha=0.35, zorder=5)
    pin_a = ax.scatter([], [], s=48, color="#be123c", edgecolors="#fff1f2", linewidths=0.8, zorder=6)
    pin_b = ax.scatter([], [], s=48, color="#be123c", edgecolors="#fff1f2", linewidths=0.8, zorder=6)

    info_box = FancyBboxPatch((0.015, 0.89), 0.43, 0.105, boxstyle="round,pad=0.02,rounding_size=0.015", transform=ax.transAxes, linewidth=0.8, edgecolor="#cbd5e1", facecolor="#ffffff", alpha=0.9)
    ax.add_patch(info_box)
    text = ax.text(0.03, 0.965, "", transform=ax.transAxes, va="top", fontsize=10, color="#0f172a")

    trace_steps = []
    if args.trace:
        trace_steps = load_trace(args.trace)

    division_steps = []
    if args.steps:
        division_steps = load_division_steps(args.steps)

    steps, mode = make_steps(points, trace_steps, args.max_frames)
    mode_label = "Divide-and-Conquer Trace" if mode == "trace" else "Best-Update Replay"

    if args.view == "dashboard":
        fig, ani = run_dashboard(points, pair, steps, point_map, args, mode_label, division_steps)
    else:
        fig, ani = run_classic(points, pair, steps, point_map, args, mode_label)

    if args.view == "dashboard":
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.90)
    else:
        plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=220)
        print(f"[viz] saved: {args.save}")

    if args.save_gif and ani is not None:
        fps = 1000.0 / 220.0 if args.view == "dashboard" else 1000.0 / 260.0
        ani.save(args.save_gif, writer=PillowWriter(fps=fps))
        print(f"[viz] gif: {args.save_gif}")

    # 默认保存模式不弹窗；仅在用户明确需要时显示窗口。
    should_show = (not args.no_show) and (not args.save) and (not args.save_gif)
    if should_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

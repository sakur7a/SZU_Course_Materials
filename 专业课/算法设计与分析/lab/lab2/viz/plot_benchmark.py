import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "n": int(row["n"]),
                    "algorithm": row["algorithm"],
                    "run_idx": int(row["run_idx"]),
                    "time_ms": float(row["time_ms"]),
                    "extrapolated": row["extrapolated"] == "1",
                }
            )
    return rows


def grouped_mean(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["n"], r["algorithm"], r["extrapolated"])
        groups[key].append(r["time_ms"])

    agg = []
    for key, vals in groups.items():
        n, algorithm, extrapolated = key
        agg.append(
            {
                "n": n,
                "algorithm": algorithm,
                "extrapolated": extrapolated,
                "mean_ms": sum(vals) / len(vals),
                "std_ms": math.sqrt(sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / len(vals)) if vals else 0.0,
                "count": len(vals),
            }
        )
    agg.sort(key=lambda x: (x["algorithm"], x["n"], x["extrapolated"]))
    return agg


def thousands(x, _):
    return f"{int(x):,}"


def apply_theme():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d2d8e2",
            "axes.labelcolor": "#1f2937",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#dbe3ef",
            "grid.alpha": 0.85,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d2d8e2",
            "font.size": 10,
        }
    )


def draw_series(ax, agg, algo, color, marker):
    measured = [x for x in agg if x["algorithm"] == algo and not x["extrapolated"]]
    if measured:
        xs = [x["n"] for x in measured]
        ys = [x["mean_ms"] for x in measured]
        es = [x["std_ms"] for x in measured]
        ax.plot(xs, ys, marker=marker, linewidth=2.4, markersize=6, color=color, label=f"{algo} (measured)")
        upper = [y + e for y, e in zip(ys, es)]
        lower = [max(0.0, y - e) for y, e in zip(ys, es)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.15)

    extrapolated = [x for x in agg if x["algorithm"] == algo and x["extrapolated"]]
    if extrapolated:
        xs = [x["n"] for x in extrapolated]
        ys = [x["mean_ms"] for x in extrapolated]
        ax.plot(xs, ys, marker=marker, linewidth=2.0, markersize=5, linestyle="--", color=color, alpha=0.85, label=f"{algo} (extrapolated)")


def add_complexity_guides(ax, agg):
    divide = sorted([x for x in agg if x["algorithm"] == "divide" and not x["extrapolated"]], key=lambda t: t["n"])
    brute = sorted([x for x in agg if x["algorithm"] == "bruteforce" and not x["extrapolated"]], key=lambda t: t["n"])

    if len(divide) >= 1:
        n0 = divide[0]["n"]
        t0 = divide[0]["mean_ms"]
        xs = [x["n"] for x in divide]
        ref = [t0 * ((n * math.log2(max(2, n))) / (n0 * math.log2(max(2, n0)))) for n in xs]
        ax.plot(xs, ref, color="#0f766e", linestyle=":", linewidth=1.8, label="reference: n log n")

    if len(brute) >= 1:
        n0 = brute[0]["n"]
        t0 = brute[0]["mean_ms"]
        xs = [x["n"] for x in brute]
        ref = [t0 * ((n * n) / (n0 * n0)) for n in xs]
        ax.plot(xs, ref, color="#7c2d12", linestyle=":", linewidth=1.8, label="reference: n^2")


def plot_linear(agg, output_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    draw_series(ax, agg, "divide", "#2563eb", "o")
    draw_series(ax, agg, "bruteforce", "#dc2626", "s")

    ax.set_title("Closest Pair Benchmark (Linear Scale)")
    ax.set_xlabel("N (number of points)")
    ax.set_ylabel("Time (ms)")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.grid(True, alpha=0.75)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_loglog(agg, output_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algo, marker, color in [("divide", "o", "#2563eb"), ("bruteforce", "s", "#dc2626")]:
        sub = sorted([x for x in agg if x["algorithm"] == algo], key=lambda t: t["n"])
        if sub:
            ax.plot(
                [x["n"] for x in sub],
                [x["mean_ms"] for x in sub],
                marker=marker,
                linewidth=2.4,
                markersize=6,
                color=color,
                label=algo,
            )

    add_complexity_guides(ax, agg)
    ax.set_title("Closest Pair Benchmark (Log-Log Scale)")
    ax.set_xlabel("N (number of points)")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.8, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", default=".")
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise ValueError("Benchmark CSV has no rows.")

    apply_theme()
    agg = grouped_mean(rows)

    linear_path = f"{args.output_prefix}/benchmark_linear.png"
    log_path = f"{args.output_prefix}/benchmark_loglog.png"

    plot_linear(agg, linear_path)
    plot_loglog(agg, log_path)

    print(f"[plot] linear: {linear_path}")
    print(f"[plot] loglog: {log_path}")


if __name__ == "__main__":
    main()

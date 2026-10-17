import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(path: str):
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["s"])
            algo = row["algorithm"]
            avg_ms = float(row["avg_ms"])
            ci95_ms = float(row["ci95_ms"])
            theory_ms = float(row["theory_ms"])
            speedup = float(row["speedup_vs_baseline"])
            data[algo]["measured"].append((s, avg_ms, ci95_ms))
            data[algo]["theory"].append((s, theory_ms))
            data[algo]["speedup"].append((s, speedup))

    for algo in data:
        data[algo]["measured"].sort(key=lambda x: x[0])
        data[algo]["theory"].sort(key=lambda x: x[0])
        data[algo]["speedup"].sort(key=lambda x: x[0])
    return data


def main():
    project_root = Path(__file__).resolve().parent.parent
    result_dir = project_root / "results" / "sparse"
    data = load_data(str(result_dir / "sparse_timing_results.csv"))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 5.5))
    for algo, series in sorted(data.items()):
        xs = [p[0] for p in series["measured"]]
        ys = [p[1] for p in series["measured"]]
        errs = [p[2] for p in series["measured"]]
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=algo)
    plt.title("Sparse Vectors: Measured Time vs s")
    plt.xlabel("Non-zero entries per user (s)")
    plt.ylabel("Average time (ms), error bar = 95% CI")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_result_measured.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for algo, series in sorted(data.items()):
        xs = [p[0] for p in series["measured"]]
        measured = [p[1] for p in series["measured"]]
        theory = [p[1] for p in series["theory"]]
        plt.plot(xs, measured, marker="o", linewidth=2, label=f"{algo} measured")
        plt.plot(xs, theory, linestyle="--", linewidth=2, label=f"{algo} theory")
    plt.title("Sparse Vectors: Measured vs Theory")
    plt.xlabel("Non-zero entries per user (s)")
    plt.ylabel("Time (ms)")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_result_theory.png", dpi=220)
    plt.close()

    baseline_speed = data["baseline_full_sort"]["speedup"]
    plt.figure(figsize=(10, 5.5))
    xs = [p[0] for p in baseline_speed]
    ys = [1.0 / p[1] if p[1] != 0 else 0.0 for p in baseline_speed]
    # 上面的baseline speedup恒为1，改为绘制优化算法相对baseline加速比
    if "optimized_topk_heap" in data:
        opt_speed = data["optimized_topk_heap"]["speedup"]
        xs = [p[0] for p in opt_speed]
        ys = [p[1] for p in opt_speed]
    plt.plot(xs, ys, marker="o", linewidth=2, color="#C4472D")
    plt.title("Sparse Vectors: Speedup of Optimized vs Baseline")
    plt.xlabel("Non-zero entries per user (s)")
    plt.ylabel("Speedup (baseline / optimized)")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_result_speedup.png", dpi=220)
    plt.close()

    print("绘图完成，已更新 results/sparse 下的图像文件")


if __name__ == "__main__":
    main()

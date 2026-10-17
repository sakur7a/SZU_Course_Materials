import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(path: str):
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["k"])
            n = int(row["n"])
            algo = row["algorithm"]
            avg_ms = float(row["avg_ms"])
            ci95_ms = float(row["ci95_ms"])
            theory_ms = float(row["theory_ms"])
            data[k][algo].append((n, avg_ms, ci95_ms, theory_ms))

    for k in data:
        for algo in data[k]:
            data[k][algo].sort(key=lambda x: x[0])
    return data


def main():
    project_root = Path(__file__).resolve().parent.parent
    result_dir = project_root / "results" / "dense"
    data = load_data(str(result_dir / "timing_results.csv"))

    plt.style.use("seaborn-v0_8-whitegrid")

    for k, algos in sorted(data.items()):
        plt.figure(figsize=(10, 5.5))
        for algo, points in sorted(algos.items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            errs = [p[2] for p in points]
            plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=algo)

        plt.title(f"Measured Time: Baseline vs Optimized (k={k}, 20 samples)")
        plt.xlabel("Number of users n")
        plt.ylabel("Average time (ms), error bar = 95% CI")
        plt.xticks(xs, rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f"result_k_{k}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5.5))
        for algo, points in sorted(algos.items()):
            xs = [p[0] for p in points]
            measured = [p[1] for p in points]
            theory = [p[3] for p in points]
            plt.plot(xs, measured, marker="o", linewidth=2, label=f"{algo} measured")
            plt.plot(xs, theory, linestyle="--", linewidth=2, label=f"{algo} theory")

        plt.title(f"Measured vs Theory (k={k}, baseline n=100000)")
        plt.xlabel("Number of users n")
        plt.ylabel("Time (ms)")
        plt.xticks(xs, rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f"result_theory_k_{k}.png", dpi=200)
        plt.close()

    print("绘图完成，已更新 results/dense 下的 result_k_*.png 和 result_theory_k_*.png")


if __name__ == "__main__":
    main()

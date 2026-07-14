import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path):
    data = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = int(row["s"])
            m = row["method"]
            data[m][s] = {
                "avg_ms": float(row["avg_ms"]),
                "ci95_ms": float(row["ci95_ms"]),
                "overlap": float(row["overlap_at_k"]),
            }
    return data


def xs_ys(data, method, key):
    xs = sorted(data[method].keys())
    ys = [data[method][x][key] for x in xs]
    return xs, ys


def main():
    root = Path(__file__).resolve().parent.parent
    result_dir = root / "results" / "advanced"
    data = load(result_dir / "sparse_lowlevel_opt_results.csv")

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 5.5))
    for method in ["baseline_aos_double", "optimized_csr_float_invnorm"]:
        xs, ys = xs_ys(data, method, "avg_ms")
        errs = [data[method][x]["ci95_ms"] for x in xs]
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=method)
    plt.title("Low-level Optimizations: Query Latency vs s")
    plt.xlabel("s")
    plt.ylabel("avg query latency (ms), 95% CI")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_lowlevel_opt_latency.png", dpi=220)
    plt.close()

    xs, ys = xs_ys(data, "optimized_csr_float_invnorm", "overlap")
    plt.figure(figsize=(10, 5.5))
    plt.plot(xs, ys, marker="o", linewidth=2, color="#C4472D")
    plt.ylim(0.0, 1.05)
    plt.title("Low-level Optimizations: Top-k Overlap vs Baseline")
    plt.xlabel("s")
    plt.ylabel("overlap@k")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_lowlevel_opt_overlap.png", dpi=220)
    plt.close()

    print("generated sparse_lowlevel_opt_latency.png / sparse_lowlevel_opt_overlap.png")


if __name__ == "__main__":
    main()

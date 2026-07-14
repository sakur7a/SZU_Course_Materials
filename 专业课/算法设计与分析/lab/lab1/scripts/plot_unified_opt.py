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
                "speedup": float(row["speedup_vs_baseline"]),
                "cands": float(row["avg_candidates"]),
                "overlap": float(row["avg_overlap_at_k"]),
            }
    return data


def series(data, method, key):
    xs = sorted(data[method].keys())
    ys = [data[method][x][key] for x in xs]
    return xs, ys


def main():
    root = Path(__file__).resolve().parent.parent
    result_dir = root / "results" / "advanced"
    data = load(result_dir / "unified_opt_results.csv")

    methods = [
        "baseline_scan_aos_double",
        "code_only_scan_csr_float",
        "algo_only_inverted_aos",
        "combined_inverted_csr_float",
    ]

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(11, 5.8))
    for m in methods:
        xs, ys = series(data, m, "avg_ms")
        errs = [data[m][x]["ci95_ms"] for x in xs]
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=m)
    plt.title("Unified Experiment: Query Latency vs s")
    plt.xlabel("s")
    plt.ylabel("avg query latency (ms), 95% CI")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "unified_opt_latency.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    for m in methods[1:]:
        xs, ys = series(data, m, "speedup")
        plt.plot(xs, ys, marker="o", linewidth=2, label=m)
    plt.title("Unified Experiment: Speedup vs Baseline")
    plt.xlabel("s")
    plt.ylabel("speedup (baseline / method)")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "unified_opt_speedup.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    for m in methods[1:]:
        xs, ys = series(data, m, "overlap")
        plt.plot(xs, ys, marker="o", linewidth=2, label=m)
    plt.ylim(0.0, 1.05)
    plt.title("Unified Experiment: Top-k Overlap vs Baseline")
    plt.xlabel("s")
    plt.ylabel("overlap@k")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "unified_opt_overlap.png", dpi=220)
    plt.close()

    print("generated unified_opt_latency.png / unified_opt_speedup.png / unified_opt_overlap.png")


if __name__ == "__main__":
    main()

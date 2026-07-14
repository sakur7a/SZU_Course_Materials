import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: str):
    data = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = int(row["s"])
            m = row["method"]
            data[m][s] = {
                "avg_ms": float(row["avg_query_ms"]),
                "ci95_ms": float(row["ci95_ms"]),
                "cands": float(row["avg_candidates"]),
                "recall": float(row["avg_recall_at_k"]),
                "idx_ms": float(row["avg_index_build_ms"]),
            }
    return data


def series(data, method, key):
    xs = sorted(data[method].keys())
    ys = [data[method][x][key] for x in xs]
    return xs, ys


def main():
    project_root = Path(__file__).resolve().parent.parent
    result_dir = project_root / "results" / "advanced"
    data = load(str(result_dir / "sparse_advanced_results.csv"))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 5.5))
    for method in ["baseline_full_scan_heap", "inverted_exact", "two_stage_recall_rerank"]:
        xs, ys = series(data, method, "avg_ms")
        errs = [data[method][x]["ci95_ms"] for x in xs]
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=method)
    plt.title("Sparse Advanced: Query Latency vs s")
    plt.xlabel("s (non-zero features per user)")
    plt.ylabel("avg query latency (ms), 95% CI")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_advanced_latency.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for method in ["baseline_full_scan_heap", "inverted_exact", "two_stage_recall_rerank"]:
        xs, ys = series(data, method, "cands")
        plt.plot(xs, ys, marker="o", linewidth=2, label=method)
    plt.title("Sparse Advanced: Candidate Count vs s")
    plt.xlabel("s")
    plt.ylabel("avg candidate count")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_advanced_candidates.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    xs, ys = series(data, "two_stage_recall_rerank", "recall")
    plt.plot(xs, ys, marker="o", linewidth=2, color="#C4472D")
    plt.ylim(0.0, 1.05)
    plt.title("Sparse Advanced: Two-stage Recall@k vs s")
    plt.xlabel("s")
    plt.ylabel("recall@k (vs inverted exact)")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(result_dir / "sparse_advanced_recall.png", dpi=220)
    plt.close()

    print("generated results/advanced 图像文件")


if __name__ == "__main__":
    main()

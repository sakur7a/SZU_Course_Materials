from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(OUT_DIR / "summary.csv")
    planar = [r for r in rows if r["name"].startswith("随机平面图")]
    maps = [r for r in rows if r["name"].endswith(".col")]

    ns = [int(r["vertices"]) for r in planar]
    times = [float(r["elapsed"]) for r in planar]
    edges = [int(r["edges"]) for r in planar]

    # --- Planar runtime (log-log) with O(n^2) reference ---
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    plt.loglog(ns, times, marker="o", linewidth=2, color="#2563eb", label="DSATUR runtime")
    # O(n^2) reference: scale so it passes through the first data point
    c_ref = times[0] / (ns[0] ** 2)
    ns_ref = [ns[0], ns[-1]]
    t_ref = [c_ref * n ** 2 for n in ns_ref]
    plt.loglog(ns_ref, t_ref, "--", color="#ef4444", linewidth=1.5, alpha=0.7, label=r"$O(n^2)$ reference")
    plt.xlabel("vertices $n$")
    plt.ylabel("time / s")
    plt.title("DSATUR 4-coloring runtime on random planar graphs")
    plt.legend()
    plt.grid(True, which="both", alpha=0.28)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "planar_runtime.png")
    plt.close()

    # --- Planar edges (log-log) with 3n-6 reference ---
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    plt.loglog(ns, edges, marker="s", linewidth=2, color="#16a34a", label="actual edges")
    ns_line = [ns[0], ns[-1]]
    m_line = [3 * n - 6 for n in ns_line]
    plt.loglog(ns_line, m_line, "--", color="#f59e0b", linewidth=1.5, alpha=0.7, label=r"$m = 3n - 6$")
    plt.xlabel("vertices $n$")
    plt.ylabel("edges $m$")
    plt.title("Random Apollonian planar graph size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.28)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "planar_edges.png")
    plt.close()

    labels = [r["name"].replace(".col", "") for r in maps]
    map_times = [float(r["elapsed"]) for r in maps]
    colors = [int(r["colors"]) for r in maps]
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    bars = plt.bar(labels, map_times, color=["#0f766e", "#7c3aed", "#dc2626"])
    plt.ylabel("seconds")
    plt.title("DIMACS map coloring runtime")
    for bar, k in zip(bars, colors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{k} colors",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "map_runtime.png")
    plt.close()

    positions = {
        0: (1.0, 2.8), 1: (2.0, 3.2), 2: (2.15, 2.35), 3: (0.8, 1.75),
        4: (2.8, 2.35), 5: (1.25, 0.9), 6: (3.35, 1.25), 7: (4.25, 2.05),
        8: (3.2, 0.35),
    }
    edges_list = [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 4), (3, 4),
        (3, 5), (4, 6), (4, 7), (5, 6), (5, 8), (6, 7), (6, 8), (7, 8),
    ]
    color_rows = read_rows(OUT_DIR / "small_image_coloring.csv")
    palette = ["#facc15", "#60a5fa", "#f87171", "#4ade80"]
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    for u, v in edges_list:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        plt.plot([x1, x2], [y1, y2], color="#64748b", linewidth=1.3, zorder=1)
    for row in color_rows:
        v = int(row["vertex"]) - 1
        c = int(row["color"]) - 1
        x, y = positions[v]
        plt.scatter([x], [y], s=720, color=palette[c], edgecolor="#111827", linewidth=1.2, zorder=2)
        plt.text(x, y, str(v + 1), ha="center", va="center", fontsize=10, weight="bold", zorder=3)
    plt.axis("off")
    plt.title("Small pasted-map adjacency abstraction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "small_graph_coloring.png")
    plt.close()


if __name__ == "__main__":
    main()

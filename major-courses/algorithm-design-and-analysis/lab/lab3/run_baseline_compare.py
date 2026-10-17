from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from src.map_coloring import (
    brute_force_backtracking,
    dsatur_coloring,
    edge_count,
    random_apollonian_planar_graph,
    small_image_graph,
    validate_coloring,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def measure(name: str, graph, k: int, brute_timeout: float = 8.0) -> dict[str, object]:
    start = time.perf_counter()
    brute_ok, brute_colors, brute_assignments = brute_force_backtracking(graph, k, timeout=brute_timeout)
    brute_elapsed = time.perf_counter() - start
    brute_conflicts = validate_coloring(graph, brute_colors) if brute_ok else ""

    start = time.perf_counter()
    dsatur_ok, dsatur_colors, dsatur_assignments, _, _ = dsatur_coloring(graph, k, timeout=8.0)
    dsatur_elapsed = time.perf_counter() - start
    dsatur_conflicts = validate_coloring(graph, dsatur_colors) if dsatur_ok else ""

    rate = brute_assignments / brute_elapsed if brute_elapsed > 0 else 0.0
    search_space = k ** len(graph)
    estimated_full_seconds = search_space / rate if rate > 0 else math.inf
    return {
        "name": name,
        "vertices": len(graph),
        "edges": edge_count(graph),
        "colors": k,
        "brute_success": brute_ok,
        "brute_elapsed": round(brute_elapsed, 6),
        "brute_assignments": brute_assignments,
        "brute_conflicts": brute_conflicts,
        "dsatur_success": dsatur_ok,
        "dsatur_elapsed": round(dsatur_elapsed, 6),
        "dsatur_assignments": dsatur_assignments,
        "dsatur_conflicts": dsatur_conflicts,
        "brute_full_space": f"{search_space:.3e}",
        "brute_full_est_seconds": f"{estimated_full_seconds:.3e}",
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cases = [("小规模粘贴地图", small_image_graph(), 4)]
    for n in [12, 16, 20, 24, 28, 32]:
        cases.append((f"随机平面图n={n}", random_apollonian_planar_graph(n, seed=3000 + n), 4))

    rows = [measure(name, graph, k) for name, graph, k in cases]
    with (OUT_DIR / "baseline_compare.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

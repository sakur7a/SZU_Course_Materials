from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.setrecursionlimit(100000)

from src.map_coloring import (
    random_apollonian_planar_graph,
    result_to_dict,
    run_exact,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    planar_rows = []

    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        graph = random_apollonian_planar_graph(n, seed=1000 + n)
        result, colors = run_exact(f"随机平面图n={n}", graph, 4, timeout=60.0)
        row = result_to_dict(result)
        planar_rows.append(row)
        log(f"n={n:>6}  edges={result.edges:>6}  elapsed={result.elapsed:.6f}s  success={result.success}")

    with (OUT_DIR / "planar_efficiency.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(planar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planar_rows)

    log("\n=== Done ===")


if __name__ == "__main__":
    main()

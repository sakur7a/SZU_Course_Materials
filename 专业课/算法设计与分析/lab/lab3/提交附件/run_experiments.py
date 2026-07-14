from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.setrecursionlimit(100000)

from src.map_coloring import (
    random_apollonian_planar_graph,
    read_dimacs_col,
    result_to_dict,
    run_exact,
    run_exact_findall,
    run_local,
    small_image_graph,
    validate_coloring,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "地图数据" / "地图数据"
OUT_DIR = ROOT / "outputs"


def log(msg: str) -> None:
    print(msg, flush=True)


def write_coloring(path: Path, colors: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex", "color"])
        for idx, color in enumerate(colors, 1):
            writer.writerow([idx, color + 1 if color >= 0 else ""])


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    results = []

    small = small_image_graph()
    small_result, small_colors = run_exact("小规模粘贴地图抽象图", small, 4, timeout=10.0)
    results.append(result_to_dict(small_result))
    write_coloring(OUT_DIR / "small_image_coloring.csv", small_colors)
    log(f"[1/4] 小规模图完成: {small_result.elapsed:.3f}s, success={small_result.success}")

    for i, (filename, k) in enumerate([
        ("le450_5a.col", 5),
        ("le450_15b.col", 15),
        ("le450_25a.col", 25),
    ]):
        graph = read_dimacs_col(DATA_DIR / filename)
        result, colors = run_local(filename, graph, k, timeout=90.0)
        if result.success:
            assert validate_coloring(graph, colors) == 0
        results.append(result_to_dict(result))
        write_coloring(OUT_DIR / f"{filename}.coloring.csv", colors)
        log(f"[2/4] TabuCol {filename} k={k} 完成: {result.elapsed:.1f}s, success={result.success}")

    milestone_rows = []
    for i, (filename, k) in enumerate([
        ("le450_5a.col", 5),
        ("le450_15b.col", 15),
        ("le450_25a.col", 25),
    ]):
        graph = read_dimacs_col(DATA_DIR / filename)
        log(f"[3/4] 开始 find_all {filename} k={k} ...")
        result, colors, milestones = run_exact_findall(f"{filename}(全部解)", graph, k, timeout=300.0)
        if result.success:
            assert validate_coloring(graph, colors) == 0
        results.append(result_to_dict(result))
        write_coloring(OUT_DIR / f"{filename}.findall.coloring.csv", colors)
        for count, t in sorted(milestones.items()):
            milestone_rows.append({"name": filename, "k": k, "solution_count": count, "elapsed": t})
        ms_str = ", ".join(f"第{c}个={t:.3f}s" for c, t in sorted(milestones.items()))
        log(f"[3/4] find_all {filename} k={k} 完成: {result.elapsed:.1f}s, solutions={result.solutions}, milestones: {ms_str or '无'}")

    if milestone_rows:
        with (OUT_DIR / "findall_milestones.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "k", "solution_count", "elapsed"])
            writer.writeheader()
            writer.writerows(milestone_rows)

    planar_rows = []
    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        graph = random_apollonian_planar_graph(n, seed=1000 + n)
        timeout = 60.0
        result, colors = run_exact(f"随机平面图n={n}", graph, 4, timeout=timeout)
        row = result_to_dict(result)
        planar_rows.append(row)
        results.append(row)
        write_coloring(OUT_DIR / f"planar_n{n}_coloring.csv", colors)
        log(f"[4/4] 平面图 n={n} 完成: {result.elapsed:.3f}s, success={result.success}")

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with (OUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with (OUT_DIR / "planar_efficiency.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(planar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planar_rows)

    log("\n=== 全部完成 ===")
    log(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

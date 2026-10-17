from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import heapq
import random
import time


Graph = list[set[int]]


@dataclass
class ColoringResult:
    name: str
    vertices: int
    edges: int
    colors: int
    success: bool
    conflicts: int
    elapsed: float
    method: str
    assignments: int = 0
    restarts: int = 0
    solutions: int = 0


def edge_count(graph: Graph) -> int:
    return sum(len(neighbors) for neighbors in graph) // 2


def read_dimacs_col(path: str | Path) -> Graph:
    n = 0
    edges: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                n = int(parts[2])
            elif parts[0] == "e":
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                edges.append((u, v))
    graph = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            graph[u].add(v)
            graph[v].add(u)
    return graph


def validate_coloring(graph: Graph, colors: list[int]) -> int:
    conflicts = 0
    for u, neighbors in enumerate(graph):
        for v in neighbors:
            if u < v and colors[u] == colors[v]:
                conflicts += 1
    return conflicts


def small_image_graph() -> Graph:
    """Adjacency abstraction for the small example graph supplied in the lab."""
    graph_dict = {
        1: [2, 3, 4, 5],
        2: [1, 3, 5],
        3: [1, 2, 5],
        4: [1, 5, 6],
        5: [1, 2, 3, 4, 7, 8],
        6: [4, 7, 9],
        7: [5, 6, 8, 9],
        8: [5, 7, 9],
        9: [6, 7, 8],
    }
    n = len(graph_dict)
    graph = [set() for _ in range(n)]
    for vertex, neighbors in graph_dict.items():
        u = vertex - 1
        for neighbor in neighbors:
            v = neighbor - 1
            if u < v:
                graph[u].add(v)
                graph[v].add(u)
    return graph


def dsatur_coloring(
    graph: Graph,
    k: int,
    timeout: float = 20.0,
    find_all: bool = False,
) -> tuple[bool, list[int], int, int, dict[int, float]]:
    n = len(graph)
    colors = [-1] * n
    uncolored = set(range(n))
    degrees = [len(neighbors) for neighbors in graph]
    domains = [set(range(k)) for _ in range(n)]
    start_time = time.perf_counter()
    deadline = start_time + timeout
    assignments = 0
    solutions = 0
    best = [-1] * n
    milestone_times: dict[int, float] = {}
    next_milestone = 1

    def choose_vertex() -> int:
        return max(
            uncolored,
            key=lambda v: (len({colors[u] for u in graph[v] if colors[u] != -1}), degrees[v]),
        )

    def ordered_colors(v: int) -> list[int]:
        counts = []
        for c in domains[v]:
            impact = sum(1 for u in graph[v] if colors[u] == -1 and c in domains[u])
            counts.append((impact, c))
        return [c for _, c in sorted(counts)]

    def search() -> bool:
        nonlocal assignments, solutions, best, next_milestone
        if time.perf_counter() > deadline:
            return False
        if not uncolored:
            solutions += 1
            best = colors[:]
            if solutions == next_milestone:
                milestone_times[next_milestone] = round(time.perf_counter() - start_time, 6)
                next_milestone *= 10
            return not find_all
        v = choose_vertex()
        uncolored.remove(v)
        saved: list[tuple[int, set[int]]] = []
        for c in ordered_colors(v):
            if any(colors[u] == c for u in graph[v]):
                continue
            colors[v] = c
            assignments += 1
            feasible = True
            for u in graph[v]:
                if colors[u] == -1 and c in domains[u]:
                    saved.append((u, domains[u].copy()))
                    domains[u].remove(c)
                    if not domains[u]:
                        feasible = False
                        break
            if feasible and search():
                return True
            for u, domain in reversed(saved):
                domains[u] = domain
            saved.clear()
            colors[v] = -1
        uncolored.add(v)
        return False

    success = search()
    return success or solutions > 0, best if solutions else colors, assignments, solutions, milestone_times


def brute_force_backtracking(
    graph: Graph,
    k: int,
    timeout: float = 20.0,
) -> tuple[bool, list[int], int]:
    n = len(graph)
    colors = [-1] * n
    assignments = 0
    deadline = time.perf_counter() + timeout

    def search(v: int) -> bool:
        nonlocal assignments
        if time.perf_counter() > deadline:
            return False
        if v == n:
            return True
        for c in range(k):
            assignments += 1
            if all(colors[u] != c for u in graph[v]):
                colors[v] = c
                if search(v + 1):
                    return True
                colors[v] = -1
        return False

    success = search(0)
    return success, colors[:], assignments


def min_conflicts_coloring(
    graph: Graph,
    k: int,
    seed: int = 20260424,
    max_restarts: int = 80,
    max_steps: int = 250_000,
    timeout: float = 60.0,
) -> tuple[bool, list[int], int, int]:
    rng = random.Random(seed)
    n = len(graph)
    degree_order = sorted(range(n), key=lambda v: len(graph[v]), reverse=True)
    deadline = time.perf_counter() + timeout
    best_colors = [-1] * n
    best_conflicts = 10**9

    for restart in range(max_restarts):
        colors = [-1] * n
        for v in degree_order:
            counts = [0] * k
            for u in graph[v]:
                if colors[u] != -1:
                    counts[colors[u]] += 1
            min_count = min(counts)
            choices = [c for c, count in enumerate(counts) if count == min_count]
            colors[v] = rng.choice(choices)

        for step in range(max_steps):
            if time.perf_counter() > deadline:
                return False, best_colors, best_conflicts, restart
            conflicted = [
                v for v in range(n)
                if any(colors[v] == colors[u] for u in graph[v])
            ]
            if not conflicted:
                return True, colors, 0, restart + 1
            current_conflicts = validate_coloring(graph, colors)
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_colors = colors[:]
            v = rng.choice(conflicted)
            scores = []
            for c in range(k):
                scores.append(sum(1 for u in graph[v] if colors[u] == c))
            min_score = min(scores)
            choices = [c for c, score in enumerate(scores) if score == min_score]
            colors[v] = rng.choice(choices)
    return False, best_colors, best_conflicts, max_restarts


def tabucol_coloring(
    graph: Graph,
    k: int,
    seed: int = 20260424,
    max_steps: int = 2_000_000,
    timeout: float = 120.0,
    tabu_base: int = 7,
) -> tuple[bool, list[int], int, int]:
    rng = random.Random(seed)
    n = len(graph)
    deadline = time.perf_counter() + timeout
    degree_order = sorted(range(n), key=lambda v: len(graph[v]), reverse=True)
    colors = [-1] * n
    for v in degree_order:
        counts = [0] * k
        for u in graph[v]:
            if colors[u] != -1:
                counts[colors[u]] += 1
        best = min(counts)
        colors[v] = rng.choice([c for c, count in enumerate(counts) if count == best])

    adj_color = [[0] * k for _ in range(n)]
    for v in range(n):
        for u in graph[v]:
            if colors[u] >= 0:
                adj_color[v][colors[u]] += 1

    conflict_edges = validate_coloring(graph, colors)
    best_colors = colors[:]
    best_conflicts = conflict_edges
    tabu_until = [[0] * k for _ in range(n)]
    restarts = 0
    stagnant = 0

    for step in range(1, max_steps + 1):
        if conflict_edges == 0:
            return True, colors, 0, restarts + 1
        if conflict_edges < best_conflicts:
            best_conflicts = conflict_edges
            best_colors = colors[:]
            stagnant = 0
        else:
            stagnant += 1
        if time.perf_counter() > deadline:
            return False, best_colors, best_conflicts, restarts + 1

        if stagnant > 150_000:
            restarts += 1
            stagnant = 0
            for v in degree_order:
                counts = [0] * k
                for u in graph[v]:
                    if colors[u] != -1:
                        counts[colors[u]] += 1
                best = min(counts)
                colors[v] = rng.choice([c for c, count in enumerate(counts) if count == best])
            adj_color = [[0] * k for _ in range(n)]
            for v in range(n):
                for u in graph[v]:
                    adj_color[v][colors[u]] += 1
            conflict_edges = validate_coloring(graph, colors)
            continue

        conflicted = [v for v in range(n) if adj_color[v][colors[v]] > 0]
        sample_size = min(len(conflicted), 80)
        candidates = rng.sample(conflicted, sample_size) if len(conflicted) > sample_size else conflicted

        best_delta = 10**9
        moves: list[tuple[int, int, int]] = []
        for v in candidates:
            old = colors[v]
            old_count = adj_color[v][old]
            for new in range(k):
                if new == old:
                    continue
                delta = adj_color[v][new] - old_count
                projected = conflict_edges + delta
                tabu = tabu_until[v][new] > step
                if tabu and projected >= best_conflicts:
                    continue
                if delta < best_delta:
                    best_delta = delta
                    moves = [(v, old, new)]
                elif delta == best_delta:
                    moves.append((v, old, new))

        if not moves:
            continue
        v, old, new = rng.choice(moves)
        conflict_edges += adj_color[v][new] - adj_color[v][old]
        colors[v] = new
        tabu_until[v][old] = step + tabu_base + rng.randrange(10) + int(0.6 * conflict_edges)
        for u in graph[v]:
            adj_color[u][old] -= 1
            adj_color[u][new] += 1

    return False, best_colors, best_conflicts, restarts + 1


def random_apollonian_planar_graph(n: int, seed: int) -> Graph:
    if n < 3:
        raise ValueError("n must be at least 3")
    rng = random.Random(seed)
    graph = [set() for _ in range(n)]
    faces: list[tuple[int, int, int]] = [(0, 1, 2)]
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        graph[u].add(v)
        graph[v].add(u)
    for new_v in range(3, n):
        idx = rng.randrange(len(faces))
        a, b, c = faces.pop(idx)
        for u in (a, b, c):
            graph[new_v].add(u)
            graph[u].add(new_v)
        faces.extend([(a, b, new_v), (a, c, new_v), (b, c, new_v)])
    return graph


def run_exact(name: str, graph: Graph, colors: int, timeout: float = 20.0) -> tuple[ColoringResult, list[int]]:
    start = time.perf_counter()
    success, coloring, assignments, _, _ = dsatur_coloring(graph, colors, timeout=timeout)
    elapsed = time.perf_counter() - start
    conflicts = validate_coloring(graph, coloring) if success else -1
    return (
        ColoringResult(name, len(graph), edge_count(graph), colors, success, conflicts, elapsed, "DSATUR回溯", assignments),
        coloring,
    )


def run_local(name: str, graph: Graph, colors: int, timeout: float = 60.0) -> tuple[ColoringResult, list[int]]:
    start = time.perf_counter()
    success, coloring, conflicts, restarts = tabucol_coloring(graph, colors, timeout=timeout)
    elapsed = time.perf_counter() - start
    if success:
        conflicts = validate_coloring(graph, coloring)
    return (
        ColoringResult(name, len(graph), edge_count(graph), colors, success, conflicts, elapsed, "TabuCol局部搜索", 0, restarts),
        coloring,
    )


def run_exact_findall(name: str, graph: Graph, colors: int, timeout: float = 120.0) -> tuple[ColoringResult, list[int], dict[int, float]]:
    start = time.perf_counter()
    success, coloring, assignments, solutions, milestone_times = dsatur_coloring(graph, colors, timeout=timeout, find_all=True)
    elapsed = time.perf_counter() - start
    conflicts = validate_coloring(graph, coloring) if success else -1
    return (
        ColoringResult(name, len(graph), edge_count(graph), colors, success, conflicts, elapsed, "DSATUR回溯(全部解)", assignments, 0, solutions),
        coloring,
        milestone_times,
    )


def result_to_dict(result: ColoringResult) -> dict[str, object]:
    return {
        "name": result.name,
        "vertices": result.vertices,
        "edges": result.edges,
        "colors": result.colors,
        "success": result.success,
        "conflicts": result.conflicts,
        "elapsed": round(result.elapsed, 6),
        "method": result.method,
        "assignments": result.assignments,
        "restarts": result.restarts,
        "solutions": result.solutions,
    }

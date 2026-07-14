"""
最大流算法优化版
包含多种优化策略的实现和基准测试

优化策略：
1. Dinic + 当前弧优化 + BFS提前终止
2. FIFO Push-Relabel + Gap启发式
3. Highest-Label Push-Relabel + Gap启发式
"""

from collections import deque
import time
from maxflow import Edge, FlowNetwork


# ─────────────────────────────────────────────
# 优化1：Dinic + 当前弧优化
# ─────────────────────────────────────────────

class DinicOptimized(FlowNetwork):
    """
    Dinic算法优化版
    优化点：
    1. 当前弧优化：跳过已确认无法增广的边
    2. BFS提前终止：找到汇点后立即停止
    """

    def __init__(self, n):
        super().__init__(n)
        self.name = "Dinic (Optimized)"

    def bfs(self, s, t):
        """BFS构建层次图（提前终止）"""
        level = [-1] * self.n
        level[s] = 0
        queue = deque([s])
        graph = self.graph

        while queue:
            u = queue.popleft()
            for e in graph[u]:
                if e.cap - e.flow > 0 and level[e.to] < 0:
                    level[e.to] = level[u] + 1
                    if e.to == t:
                        self.level = level
                        return True
                    queue.append(e.to)

        self.level = level
        return level[t] >= 0

    def dfs(self, u, t, f):
        """DFS寻找阻塞流（带当前弧优化）"""
        if u == t:
            return f

        graph_u = self.graph[u]
        iter_u = self.iter[u]
        level = self.level

        while iter_u < len(graph_u):
            e = graph_u[iter_u]
            if e.cap - e.flow > 0 and level[u] < level[e.to]:
                d = self.dfs(e.to, t, min(f, e.cap - e.flow))
                if d > 0:
                    e.flow += d
                    e.rev.flow -= d
                    self.iter[u] = iter_u
                    return d
            iter_u += 1

        self.iter[u] = iter_u
        return 0

    def max_flow(self, s, t):
        total = 0
        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                total += f
        return total


# ─────────────────────────────────────────────
# 优化2：FIFO Push-Relabel + Gap启发式
# ─────────────────────────────────────────────

class PushRelabelFIFO(FlowNetwork):
    """
    FIFO Push-Relabel + Gap启发式

    改进：
    1. 使用FIFO队列选择活跃节点
    2. 正确实现Gap启发式
    """

    def __init__(self, n):
        super().__init__(n)
        self.name = "Push-Relabel (FIFO+Gap)"

    def max_flow(self, s, t):
        n = self.n
        graph = self.graph
        height = [0] * n
        excess = [0] * n
        gap = [0] * (2 * n + 1)
        queue = deque()
        in_queue = [False] * n

        # 初始化
        height[s] = n
        gap[n] = 1
        gap[0] = n - 1

        for e in graph[s]:
            if e.cap > 0:
                amt = e.cap
                e.flow += amt
                e.rev.flow -= amt
                excess[e.to] += amt
                excess[s] -= amt
                if e.to != s and e.to != t:
                    queue.append(e.to)
                    in_queue[e.to] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            # Discharge u
            while excess[u] > 0:
                # 尝试Push
                pushed = False
                for e in graph[u]:
                    if e.cap - e.flow > 0 and height[u] == height[e.to] + 1:
                        amt = min(excess[u], e.cap - e.flow)
                        e.flow += amt
                        e.rev.flow -= amt
                        excess[u] -= amt
                        excess[e.to] += amt
                        if e.to != s and e.to != t and not in_queue[e.to]:
                            queue.append(e.to)
                            in_queue[e.to] = True
                        if excess[u] == 0:
                            pushed = True
                            break
                        pushed = True

                if excess[u] == 0:
                    break

                if not pushed:
                    # Relabel
                    old_h = height[u]
                    min_h = 2 * n
                    for e in graph[u]:
                        if e.cap - e.flow > 0 and height[e.to] < min_h:
                            min_h = height[e.to]

                    # Gap启发式
                    gap[old_h] -= 1
                    if gap[old_h] == 0 and old_h < n:
                        for v in range(n):
                            if old_h < height[v] < n:
                                gap[height[v]] -= 1
                                height[v] = n + 1
                                gap[n + 1] += 1
                                if in_queue[v]:
                                    in_queue[v] = False
                        break  # u的discharge结束

                    height[u] = min_h + 1
                    gap[height[u]] += 1

                    # 重新加入队列继续discharge
                    if not in_queue[u]:
                        queue.append(u)
                        in_queue[u] = True

        return excess[t]


# ─────────────────────────────────────────────
# 优化3：Highest-Label Push-Relabel + Gap启发式
# ─────────────────────────────────────────────

class PushRelabelHLGap(FlowNetwork):
    """
    Highest-Label Push-Relabel + Gap启发式

    理论最优变体：
    - 选择高度最高的活跃节点
    - 时间复杂度 O(V^2 * sqrt(E))
    """

    def __init__(self, n):
        super().__init__(n)
        self.name = "Push-Relabel (HL+Gap)"

    def max_flow(self, s, t):
        n = self.n
        graph = self.graph
        height = [0] * n
        excess = [0] * n
        gap = [0] * (2 * n + 1)

        # 按高度分桶（需要容纳 height = n+1 的节点）
        bucket = [[] for _ in range(2 * n + 2)]
        in_bucket = [False] * n
        max_h = 0

        def activate(v):
            nonlocal max_h
            if v != s and v != t and excess[v] > 0 and not in_bucket[v]:
                h = height[v]
                bucket[h].append(v)
                in_bucket[v] = True
                if h > max_h:
                    max_h = h

        def deactivate(v):
            in_bucket[v] = False

        # 初始化
        height[s] = n
        gap[n] = 1
        gap[0] = n - 1

        for e in graph[s]:
            if e.cap > 0:
                amt = e.cap
                e.flow += amt
                e.rev.flow -= amt
                excess[e.to] += amt
                excess[s] -= amt
                activate(e.to)

        # 主循环
        while max_h >= 0:
            while max_h >= 0 and not bucket[max_h]:
                max_h -= 1
            if max_h < 0:
                break

            u = bucket[max_h].pop()
            in_bucket[u] = False

            # Discharge
            while excess[u] > 0:
                pushed = False
                for e in graph[u]:
                    if e.cap - e.flow > 0 and height[u] == height[e.to] + 1:
                        amt = min(excess[u], e.cap - e.flow)
                        e.flow += amt
                        e.rev.flow -= amt
                        excess[u] -= amt
                        excess[e.to] += amt
                        activate(e.to)
                        if excess[u] == 0:
                            pushed = True
                            break
                        pushed = True

                if excess[u] == 0:
                    break

                if not pushed:
                    old_h = height[u]
                    min_h = 2 * n
                    for e in graph[u]:
                        if e.cap - e.flow > 0 and height[e.to] < min_h:
                            min_h = height[e.to]

                    gap[old_h] -= 1
                    if gap[old_h] == 0 and old_h < n:
                        for v in range(n):
                            if old_h < height[v] < n:
                                gap[height[v]] -= 1
                                height[v] = n + 1
                                gap[n + 1] += 1
                                deactivate(v)
                        # 将当前节点高度设为 n+1，让它可以继续 push 回源点
                        height[u] = n + 1
                        gap[n + 1] += 1
                    else:
                        height[u] = min_h + 1
                        gap[height[u]] += 1
                    activate(u)

        return excess[t]


# ─────────────────────────────────────────────
# 基准测试
# ─────────────────────────────────────────────

def benchmark_algorithms(n, edges, s, t):
    """基准测试不同算法"""
    from maxflow import FordFulkerson, EdmondsKarp, Dinic, PushRelabel

    algorithms = [
        ("Edmonds-Karp", EdmondsKarp),
        ("Dinic", Dinic),
        ("Dinic (Optimized)", DinicOptimized),
        ("Push-Relabel", PushRelabel),
        ("Push-Relabel (FIFO+Gap)", PushRelabelFIFO),
        ("Push-Relabel (HL+Gap)", PushRelabelHLGap),
    ]

    results = []
    for name, AlgoClass in algorithms:
        G = AlgoClass(n)
        for u, v, cap in edges:
            G.add_edge(u, v, cap)

        start = time.time()
        flow = G.max_flow(s, t)
        elapsed = (time.time() - start) * 1000

        results.append({
            'name': name,
            'flow': flow,
            'time': elapsed
        })

    return results


def benchmark_large_scale():
    """大规模基准测试"""
    import random
    import sys
    sys.setrecursionlimit(10000)

    print("=" * 80)
    print("大规模基准测试")
    print("=" * 80)

    from maxflow import EdmondsKarp, Dinic

    algorithms = [
        ("Edmonds-Karp", EdmondsKarp),
        ("Dinic", Dinic),
        ("Dinic (Optimized)", DinicOptimized),
        ("Push-Relabel (FIFO+Gap)", PushRelabelFIFO),
        ("Push-Relabel (HL+Gap)", PushRelabelHLGap),
    ]

    test_sizes = [50, 100, 200, 300, 500, 800, 1000]

    # 表头
    print(f"\n{'规模':>6s}", end="")
    for name, _ in algorithms:
        short = name.replace("Push-Relabel", "PR").replace("Optimized", "Opt")
        print(f" {short:>18s}", end="")
    print()
    print("-" * (6 + 19 * len(algorithms)))

    for n in test_sizes:
        random.seed(42)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    cap = random.randint(1, 20)
                    edges.append((i, j, cap))

        results = []
        for name, AlgoClass in algorithms:
            G = AlgoClass(n)
            for u, v, cap in edges:
                G.add_edge(u, v, cap)

            start = time.time()
            flow = G.max_flow(0, n - 1)
            elapsed = (time.time() - start) * 1000
            results.append((flow, elapsed))

        print(f"{n:>6d}", end="")
        for flow, t in results:
            print(f" {t:>15.2f} ms", end="")
        print()

    # 验证正确性
    print("\n正确性验证（经典6节点网络，最大流=23）:")
    test_edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 10), (1, 3, 12),
        (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20),
        (4, 3, 7), (4, 5, 4)
    ]
    for name, AlgoClass in algorithms:
        G = AlgoClass(6)
        for u, v, cap in test_edges:
            G.add_edge(u, v, cap)
        flow = G.max_flow(0, 5)
        status = "OK" if flow == 23 else f"WRONG ({flow})"
        print(f"  {name:30s}: {status}")


if __name__ == "__main__":
    benchmark_large_scale()

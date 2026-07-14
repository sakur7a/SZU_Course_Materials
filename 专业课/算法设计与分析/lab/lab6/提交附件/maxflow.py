"""
最大流算法实现
包含三种经典算法：Ford-Fulkerson、Edmonds-Karp、Dinic's Algorithm
"""

from collections import deque, defaultdict
import time


class Edge:
    """边类"""
    def __init__(self, to, cap, flow=0):
        self.to = to          # 终点
        self.cap = cap        # 容量
        self.flow = flow      # 当前流量
        self.rev = None       # 反向边引用


class FlowNetwork:
    """流网络基类"""
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []

    def add_edge(self, u, v, cap):
        """添加边"""
        forward = Edge(v, cap)
        backward = Edge(u, 0)
        forward.rev = backward
        backward.rev = forward
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        self.edges.append((u, v, cap))

    def get_flow_on_edge(self, u, v):
        """获取边(u,v)上的流量"""
        for e in self.graph[u]:
            if e.to == v:
                return e.flow
        return 0


class FordFulkerson(FlowNetwork):
    """
    Ford-Fulkerson 方法
    时间复杂度：O(E * max_flow)
    使用DFS寻找增广路径
    """
    def __init__(self, n):
        super().__init__(n)
        self.name = "Ford-Fulkerson"

    def bfs(self, s, t, parent):
        """BFS寻找增广路径（也可用DFS）"""
        visited = [False] * self.n
        queue = deque([s])
        visited[s] = True

        while queue:
            u = queue.popleft()
            for e in self.graph[u]:
                if not visited[e.to] and e.cap - e.flow > 0:
                    visited[e.to] = True
                    parent[e.to] = e
                    if e.to == t:
                        return True
                    queue.append(e.to)
        return False

    def dfs(self, u, t, min_cap, visited):
        """DFS寻找增广路径"""
        if u == t:
            return min_cap
        visited[u] = True
        for e in self.graph[u]:
            if not visited[e.to] and e.cap - e.flow > 0:
                bottleneck = min(min_cap, e.cap - e.flow)
                result = self.dfs(e.to, t, bottleneck, visited)
                if result > 0:
                    e.flow += result
                    e.rev.flow -= result
                    return result
        return 0

    def max_flow(self, s, t):
        """计算最大流"""
        parent = [-1] * self.n
        total_flow = 0

        while self.bfs(s, t, parent):
            # 找到增广路径的瓶颈容量
            path_flow = float('inf')
            v = t
            while v != s:
                e = parent[v]
                path_flow = min(path_flow, e.cap - e.flow)
                v = e.rev.to

            # 更新流量
            v = t
            while v != s:
                e = parent[v]
                e.flow += path_flow
                e.rev.flow -= path_flow
                v = e.rev.to

            total_flow += path_flow
            parent = [-1] * self.n

        return total_flow


class EdmondsKarp(FlowNetwork):
    """
    Edmonds-Karp 算法
    时间复杂度：O(V * E^2)
    使用BFS寻找最短增广路径
    """
    def __init__(self, n):
        super().__init__(n)
        self.name = "Edmonds-Karp"

    def bfs(self, s, t, parent):
        """BFS寻找最短增广路径"""
        visited = [False] * self.n
        queue = deque([s])
        visited[s] = True

        while queue:
            u = queue.popleft()
            for e in self.graph[u]:
                if not visited[e.to] and e.cap - e.flow > 0:
                    visited[e.to] = True
                    parent[e.to] = e
                    if e.to == t:
                        return True
                    queue.append(e.to)
        return False

    def max_flow(self, s, t):
        """计算最大流"""
        parent = [-1] * self.n
        total_flow = 0

        while self.bfs(s, t, parent):
            # 找到瓶颈容量
            path_flow = float('inf')
            v = t
            while v != s:
                e = parent[v]
                path_flow = min(path_flow, e.cap - e.flow)
                v = e.rev.to

            # 更新流量
            v = t
            while v != s:
                e = parent[v]
                e.flow += path_flow
                e.rev.flow -= path_flow
                v = e.rev.to

            total_flow += path_flow
            parent = [-1] * self.n

        return total_flow


class Dinic(FlowNetwork):
    """
    Dinic 算法
    时间复杂度：O(V^2 * E)
    使用层次图和阻塞流
    """
    def __init__(self, n):
        super().__init__(n)
        self.name = "Dinic"
        self.level = [0] * n
        self.iter = [0] * n

    def bfs(self, s, t):
        """BFS构建层次图"""
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])

        while queue:
            u = queue.popleft()
            for e in self.graph[u]:
                if e.cap - e.flow > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[u] + 1
                    queue.append(e.to)

        return self.level[t] >= 0

    def dfs(self, u, t, f):
        """DFS寻找阻塞流"""
        if u == t:
            return f

        for i in range(self.iter[u], len(self.graph[u])):
            e = self.graph[u][i]
            if e.cap - e.flow > 0 and self.level[u] < self.level[e.to]:
                d = self.dfs(e.to, t, min(f, e.cap - e.flow))
                if d > 0:
                    e.flow += d
                    e.rev.flow -= d
                    return d
            self.iter[u] += 1

        return 0

    def max_flow(self, s, t):
        """计算最大流"""
        total_flow = 0

        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                total_flow += f

        return total_flow


class PushRelabel(FlowNetwork):
    """
    预流推进算法 (Push-Relabel)
    时间复杂度：O(V^2 * E) 或 O(V^3)
    使用预流和高度标签
    """
    def __init__(self, n):
        super().__init__(n)
        self.name = "Push-Relabel"
        self.height = [0] * n
        self.excess = [0] * n
        self.active = [False] * n

    def push(self, e, u):
        """推流操作"""
        flow = min(self.excess[u], e.cap - e.flow)
        e.flow += flow
        e.rev.flow -= flow
        self.excess[u] -= flow
        self.excess[e.to] += flow
        if self.excess[e.to] > 0 and not self.active[e.to]:
            self.active[e.to] = True

    def relabel(self, u):
        """重新标记操作"""
        min_height = float('inf')
        for e in self.graph[u]:
            if e.cap - e.flow > 0:
                min_height = min(min_height, self.height[e.to])
        self.height[u] = min_height + 1

    def max_flow(self, s, t):
        """计算最大流"""
        # 初始化
        self.height[s] = self.n
        self.excess[s] = float('inf')

        # 从源点出发推流
        for e in self.graph[s]:
            self.push(e, s)

        # 主循环
        while True:
            # 找到活跃节点
            active_node = -1
            for i in range(self.n):
                if i != s and i != t and self.active[i]:
                    if active_node == -1 or self.height[i] > self.height[active_node]:
                        active_node = i

            if active_node == -1:
                break

            u = active_node
            pushed = False

            # 尝试推流
            for e in self.graph[u]:
                if e.cap - e.flow > 0 and self.height[u] == self.height[e.to] + 1:
                    self.push(e, u)
                    pushed = True
                    if self.excess[u] == 0:
                        self.active[u] = False
                        break

            # 如果无法推流，重新标记
            if not pushed:
                self.relabel(u)

        return self.excess[t]


def benchmark_algorithms(n, edges, s, t):
    """基准测试不同算法"""
    algorithms = [
        ("Ford-Fulkerson", FordFulkerson),
        ("Edmonds-Karp", EdmondsKarp),
        ("Dinic", Dinic),
        ("Push-Relabel", PushRelabel)
    ]

    results = []
    for name, AlgoClass in algorithms:
        G = AlgoClass(n)
        for u, v, cap in edges:
            G.add_edge(u, v, cap)

        start = time.time()
        flow = G.max_flow(s, t)
        elapsed = time.time() - start

        results.append({
            'name': name,
            'flow': flow,
            'time': elapsed
        })

    return results


if __name__ == "__main__":
    # 测试最大流算法
    print("=" * 60)
    print("最大流算法测试")
    print("=" * 60)

    # 创建测试网络
    n = 6
    s, t = 0, 5

    edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 10), (1, 3, 12),
        (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20),
        (4, 3, 7), (4, 5, 4)
    ]

    print(f"\n测试网络: {n} 个节点, 源点={s}, 汇点={t}")
    print(f"边数: {len(edges)}")

    # 测试各算法
    for name, AlgoClass in [
        ("Ford-Fulkerson", FordFulkerson),
        ("Edmonds-Karp", EdmondsKarp),
        ("Dinic", Dinic),
        ("Push-Relabel", PushRelabel)
    ]:
        G = AlgoClass(n)
        for u, v, cap in edges:
            G.add_edge(u, v, cap)

        flow = G.max_flow(s, t)
        print(f"\n{name} 算法:")
        print(f"  最大流: {flow}")

    # 基准测试
    print("\n" + "=" * 60)
    print("基准测试")
    print("=" * 60)

    results = benchmark_algorithms(n, edges, s, t)
    for r in results:
        print(f"{r['name']:20s}: 流量={r['flow']}, 时间={r['time']*1000:.3f}ms")

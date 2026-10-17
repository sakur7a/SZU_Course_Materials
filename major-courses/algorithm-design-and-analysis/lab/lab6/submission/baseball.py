"""
棒球淘汰问题 (Baseball Elimination Problem)
使用最大流算法求解
"""

from maxflow import EdmondsKarp, Dinic
from maxflow_optimized import DinicOptimized, PushRelabelHLGap
from collections import defaultdict


class BaseballElimination:
    """
    棒球淘汰问题求解器

    问题描述：
    - 有n支球队
    - 每支球队已经赢了w[i]场
    - 每支球队还剩r[i]场比赛
    - g[i][j]表示球队i和球队j之间还剩多少场比赛
    - 问哪些球队已经被淘汰（不可能获得第一名或并列第一名）
    """

    def __init__(self, teams):
        """
        初始化
        teams: 字典，键为队名，值为 (wins, remaining, games_against)
        """
        self.teams = teams
        self.n = len(teams)
        self.team_names = list(teams.keys())
        self.name_to_idx = {name: i for i, name in enumerate(self.team_names)}

        # 解析数据
        self.wins = []      # 已赢场数
        self.remaining = [] # 剩余场数
        self.games = []     # 对阵矩阵

        for name in self.team_names:
            w, r, g = teams[name]
            self.wins.append(w)
            self.remaining.append(r)
            self.games.append(g)

    def is_eliminated_simple(self, team_idx):
        """
        简单淘汰检测（trivial elimination）
        如果某队即使剩余全赢也无法超过当前最高分，则被淘汰
        """
        max_wins = self.wins[team_idx] + self.remaining[team_idx]
        for i in range(self.n):
            if i != team_idx and self.wins[i] > max_wins:
                return True, [self.team_names[i]]
        return False, []

    def build_flow_network(self, team_idx):
        """
        为指定球队构建流网络

        流网络构造：
        1. 源点 s 连接到所有比赛节点（除涉及team_idx的比赛）
        2. 比赛节点连接到对应的两支球队
        3. 所有其他球队连接到汇点 t

        节点编号：
        - 0: 源点 s
        - 1 到 m: 比赛节点 (m是比赛数量)
        - m+1 到 m+n-1: 球队节点 (不包括team_idx)
        - m+n: 汇点 t
        """
        # 计算比赛节点数量
        game_nodes = []
        for i in range(self.n):
            if i == team_idx:
                continue
            for j in range(i + 1, self.n):
                if j == team_idx:
                    continue
                if self.games[i][j] > 0:
                    game_nodes.append((i, j, self.games[i][j]))

        m = len(game_nodes)  # 比赛节点数量
        if m == 0:
            return None, game_nodes

        # 节点映射
        team_to_node = {}
        node_to_team = {}
        node_idx = m + 1
        for i in range(self.n):
            if i != team_idx:
                team_to_node[i] = node_idx
                node_to_team[node_idx] = i
                node_idx += 1

        s = 0
        t = m + self.n
        G = PushRelabelHLGap(t + 1)

        # 源点到比赛节点的边
        for idx, (i, j, games) in enumerate(game_nodes):
            game_node = idx + 1
            G.add_edge(s, game_node, games)
            # 比赛节点到球队节点的边（容量无穷大）
            G.add_edge(game_node, team_to_node[i], float('inf'))
            G.add_edge(game_node, team_to_node[j], float('inf'))

        # 球队节点到汇点的边
        max_wins = self.wins[team_idx] + self.remaining[team_idx]
        for i in range(self.n):
            if i != team_idx:
                capacity = max_wins - self.wins[i]
                if capacity < 0:
                    capacity = 0
                G.add_edge(team_to_node[i], t, capacity)

        return G, game_nodes

    def is_eliminated_flow(self, team_idx):
        """
        使用最大流判断球队是否被淘汰

        原理：
        - 如果最大流等于所有比赛的总场数，则该队未被淘汰
        - 否则，该队被淘汰
        """
        # 先检查简单淘汰
        eliminated, proof = self.is_eliminated_simple(team_idx)
        if eliminated:
            return True, proof

        # 构建流网络
        G, game_nodes = self.build_flow_network(team_idx)
        if G is None:
            return False, []

        # 计算总比赛场数
        total_games = sum(g[2] for g in game_nodes)

        # 计算最大流
        max_flow = G.max_flow(0, G.n - 1)

        # 如果最大流等于总比赛场数，说明可以分配所有比赛结果
        # 使得没有其他球队超过该队的最大可能胜场
        if max_flow == total_games:
            return False, []
        else:
            # 找到最小割，即淘汰证明
            proof = self.find_elimination_proof(G, team_idx, game_nodes)
            return True, proof

    def find_elimination_proof(self, G, team_idx, game_nodes):
        """
        找到淘汰证明（最小割）
        从源点可达的节点集合S就是证明
        """
        # BFS找从源点可达的节点
        visited = [False] * G.n
        queue = [0]
        visited[0] = True

        while queue:
            u = queue.pop(0)
            for e in G.graph[u]:
                if not visited[e.to] and e.cap - e.flow > 0:
                    visited[e.to] = True
                    queue.append(e.to)

        # 收集在S中的球队
        proof_teams = []
        m = len(game_nodes)
        for i in range(G.n):
            if visited[i] and i > m and i < G.n - 1:
                team_idx_in_list = i - m - 1
                if team_idx_in_list < len(self.team_names):
                    proof_teams.append(self.team_names[team_idx_in_list])

        return proof_teams

    def solve(self):
        """求解所有球队的淘汰情况"""
        results = {}
        for i in range(self.n):
            team_name = self.team_names[i]
            eliminated, proof = self.is_eliminated_flow(i)
            results[team_name] = {
                'eliminated': eliminated,
                'proof': proof,
                'wins': self.wins[i],
                'remaining': self.remaining[i],
                'max_possible': self.wins[i] + self.remaining[i]
            }
        return results

    def print_results(self, results):
        """打印结果"""
        print("\n" + "=" * 70)
        print("棒球淘汰问题求解结果")
        print("=" * 70)

        # 打印球队信息
        print("\n球队信息:")
        print("-" * 70)
        print(f"{'球队':10s} {'已胜':>6s} {'剩余':>6s} {'最大可能':>8s} {'状态':>8s}")
        print("-" * 70)

        for team_name in self.team_names:
            r = results[team_name]
            status = "淘汰" if r['eliminated'] else "未淘汰"
            print(f"{team_name:10s} {r['wins']:6d} {r['remaining']:6d} {r['max_possible']:8d} {status:>8s}")

        # 打印淘汰证明
        print("\n淘汰证明:")
        print("-" * 70)
        for team_name in self.team_names:
            r = results[team_name]
            if r['eliminated'] and r['proof']:
                print(f"\n{team_name} 被以下球队淘汰:")
                for proof_team in r['proof']:
                    print(f"  - {proof_team}")

        return results


def create_example_data():
    """
    创建实验中的示例数据 - 四支球队问题（来自实验图片数据）
    """
    # 四支球队：亚特兰大、费城、纽约、蒙特利尔
    # 数据来自实验文档中的图片表格
    teams = {
        'Atlanta':  (83, 8, [0, 1, 6, 1]),   # 1+6+1=8
        'Philly':   (80, 3, [1, 0, 0, 2]),   # 1+0+2=3
        'New York': (78, 6, [6, 0, 0, 0]),   # 6+0+0=6
        'Montreal': (77, 3, [1, 2, 0, 0])    # 1+2+0=3
    }
    return teams


def create_nontrivial_elimination_example():
    """
    非简单淘汰的示例（与create_example_data相同）
    Philly被淘汰：max=83 = Atlanta的83，但Atlanta和NY有6场比赛
    """
    return create_example_data()


def create_cls_example():
    """
    CLRS教材中的经典四球队示例
    展示非简单淘汰
    """
    # 球队: Detroit, Toronto, Baltimore, New York
    # Detroit: w=77, r=3, max=80 -> 被淘汰（简单淘汰，因为Baltimore有83胜）
    teams = {
        'Detroit':   (77, 3, [0, 1, 1, 1]),
        'Toronto':   (80, 3, [1, 0, 1, 1]),
        'Baltimore': (83, 0, [1, 1, 0, 0]),
        'New York':  (78, 3, [1, 1, 0, 0])
    }
    return teams


def create_five_team_example():
    """
    五支球队示例
    Team E被淘汰（非简单淘汰）
    """
    # Team E: w=75, r=10, max=85
    # 但其他球队之间的比赛会产生足够多的胜场
    teams = {
        'Team A': (83, 7, [0, 1, 2, 2, 2]),  # max=90
        'Team B': (80, 10, [1, 0, 3, 3, 3]), # max=90
        'Team C': (78, 12, [2, 3, 0, 3, 4]), # max=90
        'Team D': (76, 14, [2, 3, 3, 0, 6]), # max=90
        'Team E': (75, 10, [2, 3, 4, 6, 0])  # max=85 -> 被淘汰
    }
    return teams


if __name__ == "__main__":
    print("=" * 70)
    print("棒球淘汰问题求解器")
    print("=" * 70)

    # 示例1：四支球队（简单淘汰）
    print("\n示例1：四支球队问题（简单淘汰）")
    print("-" * 70)
    teams1 = create_example_data()
    solver1 = BaseballElimination(teams1)
    results1 = solver1.solve()
    solver1.print_results(results1)

    # 示例2：四支球队（非简单淘汰）
    print("\n\n示例2：四支球队问题（非简单淘汰）")
    print("-" * 70)
    teams2 = create_nontrivial_elimination_example()
    solver2 = BaseballElimination(teams2)
    results2 = solver2.solve()
    solver2.print_results(results2)

    # 示例3：五支球队
    print("\n\n示例3：五支球队问题")
    print("-" * 70)
    teams3 = create_five_team_example()
    solver3 = BaseballElimination(teams3)
    results3 = solver3.solve()
    solver3.print_results(results3)

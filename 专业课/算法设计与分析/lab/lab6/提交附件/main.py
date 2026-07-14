"""
棒球淘汰问题 - 主程序
综合测试和分析
"""

from maxflow import FordFulkerson, EdmondsKarp, Dinic, PushRelabel, benchmark_algorithms
from baseball import BaseballElimination, create_example_data, create_complex_example
from visualization import (
    draw_flow_network, draw_baseball_network,
    draw_elimination_results, draw_algorithm_comparison, draw_cut_visualization
)
import time


def test_maxflow_algorithms():
    """测试最大流算法"""
    print("=" * 70)
    print("第一部分：最大流算法测试")
    print("=" * 70)

    # 测试网络1：经典示例
    print("\n1.1 经典测试网络")
    print("-" * 70)

    n = 6
    s, t = 0, 5
    edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 10), (1, 3, 12),
        (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20),
        (4, 3, 7), (4, 5, 4)
    ]

    print(f"节点数: {n}, 源点: {s}, 汇点: {t}")
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

        start = time.time()
        flow = G.max_flow(s, t)
        elapsed = time.time() - start

        print(f"\n{name} 算法:")
        print(f"  最大流: {flow}")
        print(f"  运行时间: {elapsed*1000:.3f} ms")

    # 测试网络2：较大网络
    print("\n\n1.2 较大测试网络")
    print("-" * 70)

    n2 = 10
    s2, t2 = 0, 9
    edges2 = [
        (0, 1, 10), (0, 2, 8), (0, 3, 5),
        (1, 4, 7), (1, 5, 6),
        (2, 4, 5), (2, 5, 4), (2, 6, 3),
        (3, 5, 3), (3, 6, 4),
        (4, 7, 8), (4, 8, 5),
        (5, 7, 6), (5, 8, 7),
        (6, 8, 9),
        (7, 9, 12),
        (8, 9, 10)
    ]

    print(f"节点数: {n2}, 源点: {s2}, 汇点: {t2}")
    print(f"边数: {len(edges2)}")

    results = benchmark_algorithms(n2, edges2, s2, t2)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  最大流: {r['flow']}")
        print(f"  运行时间: {r['time']*1000:.3f} ms")

    return results


def test_baseball_elimination():
    """测试棒球淘汰问题"""
    print("\n\n" + "=" * 70)
    print("第二部分：棒球淘汰问题求解")
    print("=" * 70)

    # 示例1：四支球队
    print("\n2.1 四支球队问题")
    print("-" * 70)
    teams1 = create_example_data()
    solver1 = BaseballElimination(teams1)
    results1 = solver1.solve()
    solver1.print_results(results1)

    # 示例2：五支球队
    print("\n\n2.2 五支球队问题")
    print("-" * 70)
    teams2 = create_complex_example()
    solver2 = BaseballElimination(teams2)
    results2 = solver2.solve()
    solver2.print_results(results2)

    return results1, results2


def analyze_flow_network():
    """分析流网络构造"""
    print("\n\n" + "=" * 70)
    print("第三部分：流网络构造分析")
    print("=" * 70)

    teams = create_example_data()
    solver = BaseballElimination(teams)

    # 为蒙特利尔队构建流网络
    team_idx = 0  # Montreal
    team_name = solver.team_names[team_idx]

    print(f"\n分析球队: {team_name}")
    print(f"已胜: {solver.wins[team_idx]}, 剩余: {solver.remaining[team_idx]}")
    print(f"最大可能胜场: {solver.wins[team_idx] + solver.remaining[team_idx]}")

    G, game_nodes = solver.build_flow_network(team_idx)

    print(f"\n流网络构造:")
    print(f"  节点数: {G.n}")
    print(f"  比赛节点数: {len(game_nodes)}")

    # 打印比赛节点
    print(f"\n比赛节点:")
    for idx, (i, j, games) in enumerate(game_nodes):
        print(f"  节点{idx+1}: {solver.team_names[i]} vs {solver.team_names[j]}: {games}场")

    # 计算最大流
    max_flow = G.max_flow(0, G.n - 1)
    total_games = sum(g[2] for g in game_nodes)

    print(f"\n最大流计算:")
    print(f"  总比赛场数: {total_games}")
    print(f"  最大流: {max_flow}")
    print(f"  是否相等: {max_flow == total_games}")

    return G, game_nodes


def run_benchmark():
    """运行基准测试"""
    print("\n\n" + "=" * 70)
    print("第四部分：算法性能基准测试")
    print("=" * 70)

    # 不同规模的测试
    test_cases = [
        ("小型网络 (6节点)", 6, 10),
        ("中型网络 (20节点)", 20, 50),
        ("大型网络 (50节点)", 50, 150),
    ]

    all_results = []

    for name, n, num_edges in test_cases:
        print(f"\n{name}")
        print("-" * 50)

        # 生成随机边
        import random
        random.seed(42)
        edges = []
        for _ in range(num_edges):
            u = random.randint(0, n-2)
            v = random.randint(u+1, n-1)
            cap = random.randint(1, 20)
            edges.append((u, v, cap))

        s, t = 0, n-1
        results = benchmark_algorithms(n, edges, s, t)

        for r in results:
            print(f"  {r['name']:20s}: 流量={r['flow']:6d}, 时间={r['time']*1000:.3f} ms")

        all_results.append({
            'test': name,
            'results': results
        })

    return all_results


def generate_visualizations():
    """生成可视化图表"""
    print("\n\n" + "=" * 70)
    print("第五部分：生成可视化图表")
    print("=" * 70)

    teams = create_example_data()
    solver = BaseballElimination(teams)
    results = solver.solve()

    # 1. 淘汰结果图
    print("\n生成淘汰结果图...")
    draw_elimination_results(results, save_path="elimination_results.png")

    # 2. 流网络图
    print("生成流网络图...")
    team_idx = 0  # Montreal
    G, game_nodes = solver.build_flow_network(team_idx)
    draw_baseball_network(solver, team_idx, G, game_nodes, save_path="flow_network.png")

    # 3. 最小割图
    print("生成最小割图...")
    draw_cut_visualization(G, 0, G.n-1,
                          title="棒球淘汰问题 - 最小割",
                          save_path="min_cut.png")

    print("\n图表已保存到当前目录")


def main():
    """主函数"""
    print("棒球淘汰问题 - 综合测试与分析")
    print("=" * 70)

    # 1. 测试最大流算法
    algo_results = test_maxflow_algorithms()

    # 2. 测试棒球淘汰问题
    baseball_results = test_baseball_elimination()

    # 3. 分析流网络构造
    flow_results = analyze_flow_network()

    # 4. 运行基准测试
    benchmark_results = run_benchmark()

    # 5. 生成可视化
    try:
        generate_visualizations()
    except Exception as e:
        print(f"\n可视化生成失败: {e}")
        print("请确保安装了 matplotlib 和 networkx")

    # 总结
    print("\n\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n1. 最大流算法实现:")
    print("   - Ford-Fulkerson 方法")
    print("   - Edmonds-Karp 算法")
    print("   - Dinic 算法")
    print("   - Push-Relabel 算法")
    print("\n2. 棒球淘汰问题:")
    print("   - 简单淘汰检测")
    print("   - 最大流淘汰检测")
    print("   - 淘汰证明生成")
    print("\n3. 可视化:")
    print("   - 流网络图")
    print("   - 淘汰结果图")
    print("   - 最小割图")


if __name__ == "__main__":
    main()

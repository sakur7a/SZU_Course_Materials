"""
鸡蛋掉落问题 - 可视化分析
生成性能对比图表
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入算法实现
from egg_drop import (
    egg_drop_dp,
    egg_drop_dp_optimized,
    egg_drop_dp_alt,
    egg_drop_memo
)


def plot_performance_comparison():
    """
    绘制不同算法的性能对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('鸡蛋掉落问题 - 性能分析', fontsize=16)

    # 1. 不同鸡蛋数下的性能
    ax1 = axes[0, 0]
    eggs_list = [2, 3, 5, 10, 20, 50]
    floors_list = [100, 500, 1000]

    for floors in floors_list:
        times = []
        for eggs in eggs_list:
            start = time.perf_counter()
            egg_drop_dp_optimized(eggs, floors)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        ax1.plot(eggs_list, times, marker='o', label=f'{floors}层')

    ax1.set_xlabel('鸡蛋数')
    ax1.set_ylabel('时间 (ms)')
    ax1.set_title('不同鸡蛋数下的性能')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 不同楼层数下的性能
    ax2 = axes[0, 1]
    floors_range = [100, 200, 500, 1000, 2000, 5000]
    eggs_fixed = [2, 5, 10]

    for eggs in eggs_fixed:
        times = []
        for floors in floors_range:
            start = time.perf_counter()
            egg_drop_dp_optimized(eggs, floors)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        ax2.plot(floors_range, times, marker='s', label=f'{eggs}个鸡蛋')

    ax2.set_xlabel('楼层数')
    ax2.set_ylabel('时间 (ms)')
    ax2.set_title('不同楼层数下的性能')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 算法对比
    ax3 = axes[1, 0]
    eggs, floors = 10, 500

    methods = ['基础DP', '二分优化DP', '交替DP']
    times = []

    # 基础DP
    start = time.perf_counter()
    egg_drop_dp(eggs, floors)
    times.append((time.perf_counter() - start) * 1000)

    # 二分优化DP
    start = time.perf_counter()
    egg_drop_dp_optimized(eggs, floors)
    times.append((time.perf_counter() - start) * 1000)

    # 交替DP
    start = time.perf_counter()
    egg_drop_dp_alt(eggs, floors)
    times.append((time.perf_counter() - start) * 1000)

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax3.bar(methods, times, color=colors)
    ax3.set_ylabel('时间 (ms)')
    ax3.set_title(f'算法性能对比 ({eggs}个鸡蛋, {floors}层楼)')

    # 添加数值标签
    for bar, time_val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{time_val:.2f}ms', ha='center', va='bottom')

    # 4. 时间增长趋势
    ax4 = axes[1, 1]
    floors_test = [100, 200, 400, 800, 1600, 3200]
    times_test = []

    for floors in floors_test:
        start = time.perf_counter()
        egg_drop_dp_optimized(2, floors)
        elapsed = (time.perf_counter() - start) * 1000
        times_test.append(elapsed)

    ax4.plot(floors_test, times_test, marker='D', color='#96ceb4', linewidth=2)
    ax4.set_xlabel('楼层数')
    ax4.set_ylabel('时间 (ms)')
    ax4.set_title('时间增长趋势 (2个鸡蛋)')
    ax4.grid(True, alpha=0.3)

    # 拟合曲线
    z = np.polyfit(np.log(floors_test), np.log(times_test), 1)
    p = np.poly1d(z)
    ax4.plot(floors_test, np.exp(p(np.log(floors_test))), '--', color='#ff6b6b',
             label=f'拟合: O(f^{z[0]:.2f})')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("性能分析图已保存: performance_analysis.png")


def plot_dp_table():
    """
    绘制DP表的热力图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('DP表热力图 (最少试验次数)', fontsize=14)

    for idx, eggs in enumerate([2, 3, 5]):
        floors = 30
        dp = [[0] * (floors + 1) for _ in range(eggs + 1)]

        for j in range(1, floors + 1):
            dp[1][j] = j

        for i in range(1, eggs + 1):
            dp[i][0] = 0
            dp[i][1] = 1

        for i in range(2, eggs + 1):
            for j in range(2, floors + 1):
                dp[i][j] = float('inf')
                for x in range(1, j + 1):
                    cost = 1 + max(dp[i - 1][x - 1], dp[i][j - x])
                    dp[i][j] = min(dp[i][j], cost)

        # 绘制热力图
        data = np.array([row[1:] for row in dp[1:]])
        im = axes[idx].imshow(data, cmap='YlOrRd', aspect='auto')
        axes[idx].set_xlabel('楼层数')
        axes[idx].set_ylabel('鸡蛋数')
        axes[idx].set_title(f'{eggs}个鸡蛋')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/dp_table_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("DP表热力图已保存: dp_table_heatmap.png")


def plot_space_analysis():
    """
    绘制空间使用分析图
    """
    import tracemalloc

    fig, ax = plt.subplots(figsize=(10, 6))

    test_cases = [
        (2, 100), (2, 500), (2, 1000),
        (5, 100), (5, 500), (5, 1000),
        (10, 100), (10, 500), (10, 1000),
    ]

    labels = [f'{e}蛋,{f}层' for e, f in test_cases]
    actual_space = []
    theoretical_space = []

    for eggs, floors in test_cases:
        tracemalloc.start()
        egg_drop_dp_optimized(eggs, floors)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        actual_space.append(peak / 1024)
        theoretical_space.append((eggs + 1) * (floors + 1) * 8 / 1024)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, actual_space, width, label='实际空间', color='#4ecdc4')
    bars2 = ax.bar(x + width/2, theoretical_space, width, label='理论空间', color='#ff6b6b')

    ax.set_xlabel('测试用例')
    ax.set_ylabel('空间 (KB)')
    ax.set_title('空间使用分析')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/space_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("空间分析图已保存: space_analysis.png")


if __name__ == '__main__':
    print("开始生成可视化图表...")
    plot_performance_comparison()
    plot_dp_table()
    plot_space_analysis()
    print("所有图表生成完成！")

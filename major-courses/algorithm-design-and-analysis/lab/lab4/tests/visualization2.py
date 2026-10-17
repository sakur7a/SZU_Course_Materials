"""
可视化图表 - 边界分析 + 算法对比
"""

import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from egg_drop import egg_drop_dp, egg_drop_dp_optimized, egg_drop_dp_alt


def plot_boundary_eggs():
    """
    边界分析：鸡蛋数 vs 答案
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：不同楼层数下，鸡蛋数对答案的影响
    ax1 = axes[0]
    floors_list = [100, 500, 1000, 5000]
    eggs_range = range(1, 21)

    for floors in floors_list:
        answers = [egg_drop_dp_alt(e, floors) for e in eggs_range]
        ax1.plot(list(eggs_range), answers, marker='o', markersize=3, label=f'{floors}层')

    ax1.set_xlabel('鸡蛋数')
    ax1.set_ylabel('最少试验次数')
    ax1.set_title('鸡蛋数对答案的影响')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 20)

    # 右图：边际收益（每增加1个鸡蛋减少的试验次数）
    ax2 = axes[1]
    floors = 1000
    eggs_range = range(1, 16)
    answers = [egg_drop_dp_alt(e, floors) for e in eggs_range]
    marginal = [answers[i-1] - answers[i] if i > 0 else 0 for i in range(len(answers))]

    colors = ['green' if v > 0 else 'gray' for v in marginal[1:]]
    ax2.bar(list(eggs_range)[1:], marginal[1:], color=colors, alpha=0.7)
    ax2.set_xlabel('鸡蛋数')
    ax2.set_ylabel('减少的试验次数')
    ax2.set_title(f'边际收益分析 (楼层数={floors})')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/boundary_eggs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("boundary_eggs.png saved")


def plot_boundary_floors():
    """
    边界分析：楼层数 vs 答案（对数关系）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    eggs_list = [1, 2, 3, 5, 10]
    floors_range = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

    for eggs in eggs_list:
        answers = [egg_drop_dp_alt(eggs, f) for f in floors_range]
        ax.plot(floors_range, answers, marker='s', markersize=5, label=f'{eggs}个鸡蛋')

    # 添加二分查找参考线
    binary_line = [math.ceil(math.log2(f+1)) for f in floors_range]
    ax.plot(floors_range, binary_line, '--', color='gray', linewidth=2, label='二分查找下限')

    # 添加线性参考线（1个鸡蛋）
    ax.plot(floors_range, floors_range, ':', color='red', linewidth=1, label='线性 (1个鸡蛋)')

    ax.set_xlabel('楼层数')
    ax.set_ylabel('最少试验次数')
    ax.set_title('楼层数与答案的关系（对数坐标）')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/boundary_floors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("boundary_floors.png saved")


def plot_algorithm_comparison():
    """
    算法对比：最优解 vs 二分查找
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左图：不同鸡蛋数下的对比
    ax1 = axes[0]
    floors = 1000
    eggs_range = [1, 2, 3, 5, 10, 20, 50]
    optimal = [egg_drop_dp_alt(e, floors) for e in eggs_range]
    binary = [math.ceil(math.log2(floors + 1))] * len(eggs_range)

    x = np.arange(len(eggs_range))
    width = 0.35

    ax1.bar(x - width/2, optimal, width, label='最优解', color='#4ecdc4')
    ax1.bar(x + width/2, binary, width, label='二分查找', color='#ff6b6b', alpha=0.7)
    ax1.set_xlabel('鸡蛋数')
    ax1.set_ylabel('试验次数')
    ax1.set_title(f'最优解 vs 二分查找 (楼层数={floors})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(eggs_range)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 中图：差异分析
    ax2 = axes[1]
    diff = [o - b for o, b in zip(optimal, binary)]
    colors = ['#ff6b6b' if d > 0 else '#4ecdc4' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.8)
    ax2.set_xlabel('鸡蛋数')
    ax2.set_ylabel('最优解 - 二分查找')
    ax2.set_title('最优解与二分查找的差异')
    ax2.set_xticks(x)
    ax2.set_xticklabels(eggs_range)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # 右图：不同楼层数下的对比
    ax3 = axes[2]
    eggs = 2
    floors_range = [10, 50, 100, 500, 1000, 5000]
    optimal = [egg_drop_dp_alt(eggs, f) for f in floors_range]
    binary = [math.ceil(math.log2(f+1)) for f in floors_range]

    x = np.arange(len(floors_range))
    ax3.bar(x - width/2, optimal, width, label='最优解', color='#4ecdc4')
    ax3.bar(x + width/2, binary, width, label='二分查找', color='#ff6b6b', alpha=0.7)
    ax3.set_xlabel('楼层数')
    ax3.set_ylabel('试验次数')
    ax3.set_title(f'最优解 vs 二分查找 (鸡蛋数={eggs})')
    ax3.set_xticks(x)
    ax3.set_xticklabels(floors_range, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("algorithm_comparison.png saved")


def plot_greedy_vs_optimal():
    """
    贪心策略 vs 最优策略 详细对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：2个鸡蛋，不同楼层数
    ax1 = axes[0]
    floors_range = [10, 20, 50, 100, 200, 500, 1000]
    optimal = [egg_drop_dp_alt(2, f) for f in floors_range]
    binary = [math.ceil(math.log2(f+1)) for f in floors_range]
    ratio = [o / b for o, b in zip(optimal, binary)]

    ax1.plot(floors_range, ratio, marker='o', color='#e74c3c', linewidth=2, markersize=8)
    ax1.axhline(y=1, color='gray', linestyle='--', label='二分查找基准')
    ax1.set_xlabel('楼层数')
    ax1.set_ylabel('最优解 / 二分查找')
    ax1.set_title('最优解与二分查找的比值 (2个鸡蛋)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：不同鸡蛋数下的比值
    ax2 = axes[1]
    floors = 1000
    eggs_range = [1, 2, 3, 5, 10, 20, 50]
    optimal = [egg_drop_dp_alt(e, floors) for e in eggs_range]
    binary = math.ceil(math.log2(floors + 1))
    ratio = [o / binary for o in optimal]

    bars = ax2.bar(range(len(eggs_range)), ratio, color='#3498db', alpha=0.8)
    ax2.axhline(y=1, color='gray', linestyle='--', label='二分查找基准')
    ax2.set_xlabel('鸡蛋数')
    ax2.set_ylabel('最优解 / 二分查找')
    ax2.set_title(f'不同鸡蛋数的比值 (楼层数={floors})')
    ax2.set_xticks(range(len(eggs_range)))
    ax2.set_xticklabels(eggs_range)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, r in zip(bars, ratio):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{r:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/greedy_vs_optimal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("greedy_vs_optimal.png saved")


def plot_summary():
    """
    综合总结图
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建一个表格形式的总结
    ax.axis('off')

    # 表格数据
    data = [
        ['算法', '时间复杂度', '空间复杂度', '适用场景', '10蛋1000层耗时'],
        ['基础DP', 'O(e×f²)', 'O(e×f)', '教学演示', '~1000ms'],
        ['记忆化搜索', 'O(e×f²)', 'O(e×f)', '递归思维', '~1000ms'],
        ['二分优化DP', 'O(e×f×log(f))', 'O(e×f)', '一般场景', '~3ms'],
        ['交替DP', 'O(e×answer)', 'O(e)', '最优解法', '~0.007ms'],
        ['组合数方法', 'O(answer×min(e,answer))', 'O(1)', '数学推导', '~0.01ms'],
    ]

    # 创建表格
    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.15, 0.2, 0.2])

    # 设置样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # 设置表头颜色
    for j in range(len(data[0])):
        table[0, j].set_facecolor('#3498db')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 设置交替行颜色
    for i in range(1, len(data)):
        for j in range(len(data[0])):
            if i % 2 == 0:
                table[i, j].set_facecolor('#ecf0f1')

    # 高亮最优解法
    for j in range(len(data[0])):
        table[4, j].set_facecolor('#2ecc71')
        table[4, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('鸡蛋掉落问题 - 算法对比总结', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('C:/Users/28068/Desktop/算设/lab4/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("summary_table.png saved")


if __name__ == '__main__':
    print("生成可视化图表...")
    plot_boundary_eggs()
    plot_boundary_floors()
    plot_algorithm_comparison()
    plot_greedy_vs_optimal()
    plot_summary()
    print("所有图表生成完成！")

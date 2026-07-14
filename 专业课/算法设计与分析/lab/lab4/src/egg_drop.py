"""
鸡蛋掉落问题 - 动态规划解法
实验四：动态规划算法设计

功能：
1. 动态规划解法（自底向上）
2. 记忆化搜索解法（自顶向下）
3. 蛮力法验证正确性
4. 性能测试与分析
"""

import time
import random
import sys
import io
from functools import lru_cache
from typing import Tuple, List
import json

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 增加递归深度限制
sys.setrecursionlimit(10000)


# ============================================================
# 方法1：动态规划（自底向上）
# ============================================================
def egg_drop_dp(eggs: int, floors: int) -> int:
    """
    动态规划解法（自底向上）

    状态定义：dp[i][j] 表示 i 个鸡蛋、j 层楼时的最少试验次数

    状态转移方程：
    dp[i][j] = min_{1<=x<=j} { max(dp[i-1][x-1], dp[i][j-x]) + 1 }

    其中：
    - dp[i-1][x-1]：鸡蛋在第 x 层碎了，需要在 x-1 层楼中用 i-1 个鸡蛋继续测试
    - dp[i][j-x]：鸡蛋在第 x 层没碎，需要在 j-x 层楼中用 i 个鸡蛋继续测试

    时间复杂度：O(e * f^2)
    空间复杂度：O(e * f)
    """
    if eggs <= 0 or floors <= 0:
        return 0
    if eggs == 1:
        return floors

    # dp[i][j]：i 个鸡蛋，j 层楼
    dp = [[0] * (floors + 1) for _ in range(eggs + 1)]

    # 初始化：1 个鸡蛋时，需要逐层尝试
    for j in range(1, floors + 1):
        dp[1][j] = j

    # 初始化：任意鸡蛋数，0 层楼需要 0 次，1 层楼需要 1 次
    for i in range(1, eggs + 1):
        dp[i][0] = 0
        dp[i][1] = 1

    # 填表
    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            dp[i][j] = float('inf')
            for x in range(1, j + 1):
                # 在第 x 层扔鸡蛋
                # 碎了：检查下方 x-1 层，用 i-1 个鸡蛋
                # 没碎：检查上方 j-x 层，用 i 个鸡蛋
                cost = 1 + max(dp[i - 1][x - 1], dp[i][j - x])
                dp[i][j] = min(dp[i][j], cost)

    return dp[eggs][floors]


# ============================================================
# 方法2：记忆化搜索（自顶向下）
# ============================================================
def egg_drop_memo(eggs: int, floors: int) -> int:
    """
    记忆化搜索解法（自顶向下）

    与自底向上相同的递推关系，但通过递归+缓存实现
    """
    memo = {}

    def solve(e: int, f: int) -> int:
        if e <= 0 or f <= 0:
            return 0
        if e == 1:
            return f
        if f == 1:
            return 1
        if (e, f) in memo:
            return memo[(e, f)]

        result = float('inf')
        for x in range(1, f + 1):
            cost = 1 + max(solve(e - 1, x - 1), solve(e, f - x))
            result = min(result, cost)

        memo[(e, f)] = result
        return result

    return solve(eggs, floors)


# ============================================================
# 方法3：二分查找优化的动态规划
# ============================================================
def egg_drop_dp_optimized(eggs: int, floors: int) -> int:
    """
    二分查找优化的动态规划

    观察：对于固定的 i 和 j，随着 x 增大：
    - dp[i-1][x-1] 单调递增
    - dp[i][j-x] 单调递减

    因此可以用二分查找找到最优的 x，使得两者的最大值最小

    时间复杂度：O(e * f * log(f))
    空间复杂度：O(e * f)
    """
    if eggs <= 0 or floors <= 0:
        return 0
    if eggs == 1:
        return floors

    dp = [[0] * (floors + 1) for _ in range(eggs + 1)]

    for j in range(1, floors + 1):
        dp[1][j] = j

    for i in range(1, eggs + 1):
        dp[i][0] = 0
        dp[i][1] = 1

    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            dp[i][j] = float('inf')
            # 二分查找最优的 x
            lo, hi = 1, j
            while lo <= hi:
                mid = (lo + hi) // 2
                cost_break = dp[i - 1][mid - 1]
                cost_not_break = dp[i][j - mid]
                cost = 1 + max(cost_break, cost_not_break)
                dp[i][j] = min(dp[i][j], cost)

                if cost_break < cost_not_break:
                    lo = mid + 1
                elif cost_break > cost_not_break:
                    hi = mid - 1
                else:
                    break

    return dp[eggs][floors]


# ============================================================
# 方法4：另一种DP状态定义
# ============================================================
def egg_drop_dp_alt(eggs: int, floors: int) -> int:
    """
    另一种状态定义方式

    状态定义：dp[i][j] 表示 i 次试验、j 个鸡蛋最多能确定的楼层数

    状态转移方程：
    dp[i][j] = dp[i-1][j-1] + dp[i-1][j] + 1

    含义：在某一层扔鸡蛋
    - 碎了：能确定 dp[i-1][j-1] 层（下方）
    - 没碎：能确定 dp[i-1][j] 层（上方）
    - 加上当前这一层

    找到最小的 i 使得 dp[i][eggs] >= floors

    时间复杂度：O(eggs * answer)
    空间复杂度：O(eggs * answer)
    """
    if eggs <= 0 or floors <= 0:
        return 0
    if eggs == 1:
        return floors

    # dp[j] 表示 i 次试验、j 个鸡蛋能确定的最大楼层数
    dp = [0] * (eggs + 1)

    trials = 0
    while dp[eggs] < floors:
        trials += 1
        # 从后往前更新，避免覆盖
        for j in range(eggs, 0, -1):
            dp[j] = dp[j] + dp[j - 1] + 1

    return trials


# ============================================================
# 方法5：组合数方法（数学方法）
# ============================================================
def egg_drop_math(eggs: int, floors: int) -> int:
    """
    组合数方法

    原理：n次试验、e个鸡蛋能确定的最大楼层数 = sum(C(n, k) for k=1..e)

    找到最小的n使得这个和 >= floors

    时间复杂度：O(answer * min(eggs, answer))
    空间复杂度：O(1)
    """
    if eggs <= 0 or floors <= 0:
        return 0
    if eggs == 1:
        return floors

    n = 1
    while True:
        total = 0
        for k in range(1, min(eggs, n) + 1):
            # 计算 C(n, k)
            c = 1
            for i in range(k):
                c = c * (n - i) // (i + 1)
            total += c
            # 提前退出：C(n,k)均为正数，部分和>=f 则完全和必然>=f
            if total >= floors:
                return n
        n += 1


# ============================================================
# 蛮力法（用于验证正确性）
# ============================================================
def egg_drop_brute_force(eggs: int, floors: int) -> int:
    """
    蛮力法：递归枚举所有可能的策略

    仅适用于小规模数据（floors <= 20, eggs <= 5）
    """
    memo = {}

    def solve(e: int, f: int) -> int:
        if e <= 0 or f <= 0:
            return 0
        if e == 1:
            return f
        if f == 1:
            return 1
        if (e, f) in memo:
            return memo[(e, f)]

        result = float('inf')
        # 枚举每一层
        for x in range(1, f + 1):
            # 最坏情况：碎了或没碎中取较大值
            worst = 1 + max(solve(e - 1, x - 1), solve(e, f - x))
            result = min(result, worst)

        memo[(e, f)] = result
        return result

    return solve(eggs, floors)


# ============================================================
# 正确性验证
# ============================================================
def verify_correctness():
    """
    验证各算法的正确性
    随机生成小规模数据，比较各算法结果
    """
    print("=" * 60)
    print("正确性验证")
    print("=" * 60)

    test_cases = [
        (1, 1), (1, 5), (1, 10),
        (2, 1), (2, 2), (2, 5), (2, 10), (2, 20),
        (3, 1), (3, 5), (3, 10), (3, 15), (3, 20),
        (4, 10), (4, 15), (4, 20),
        (5, 10), (5, 15), (5, 20),
    ]

    print(f"\n{'鸡蛋数':<8}{'楼层数':<8}{'DP':<8}{'记忆化':<8}{'优化DP':<8}{'交替DP':<8}{'蛮力法':<8}{'一致':<6}")
    print("-" * 60)

    all_pass = True
    for eggs, floors in test_cases:
        r1 = egg_drop_dp(eggs, floors)
        r2 = egg_drop_memo(eggs, floors)
        r3 = egg_drop_dp_optimized(eggs, floors)
        r4 = egg_drop_dp_alt(eggs, floors)
        r5 = egg_drop_brute_force(eggs, floors)

        consistent = r1 == r2 == r3 == r4 == r5
        if not consistent:
            all_pass = False

        print(f"{eggs:<8}{floors:<8}{r1:<8}{r2:<8}{r3:<8}{r4:<8}{r5:<8}{'PASS' if consistent else 'FAIL':<6}")

    # 随机测试
    print("\n随机测试（小规模）：")
    random.seed(42)
    for _ in range(20):
        eggs = random.randint(1, 5)
        floors = random.randint(1, 20)
        r1 = egg_drop_dp(eggs, floors)
        r2 = egg_drop_brute_force(eggs, floors)
        consistent = r1 == r2
        if not consistent:
            all_pass = False
        print(f"  eggs={eggs}, floors={floors}: DP={r1}, brute={r2}, {'PASS' if consistent else 'FAIL'}")

    print(f"\nVerification: {'ALL PASSED' if all_pass else 'INCONSISTENT'}")
    return all_pass


# ============================================================
# 性能测试
# ============================================================
def performance_test():
    """
    测试各算法在不同数据规模下的性能
    """
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    # 测试不同规模
    test_configs = [
        # (eggs, floors, description)
        (2, 100, "小规模 (2鸡蛋, 100层)"),
        (2, 500, "中规模 (2鸡蛋, 500层)"),
        (2, 1000, "大规模 (2鸡蛋, 1000层)"),
        (3, 100, "小规模 (3鸡蛋, 100层)"),
        (3, 500, "中规模 (3鸡蛋, 500层)"),
        (5, 100, "小规模 (5鸡蛋, 100层)"),
        (5, 500, "中规模 (5鸡蛋, 500层)"),
        (10, 100, "小规模 (10鸡蛋, 100层)"),
        (10, 500, "中规模 (10鸡蛋, 500层)"),
        (10, 1000, "大规模 (10鸡蛋, 1000层)"),
        (10, 2000, "大规模 (10鸡蛋, 2000层)"),
        (20, 1000, "大规模 (20鸡蛋, 1000层)"),
        (50, 1000, "大规模 (50鸡蛋, 1000层)"),
        (100, 1000, "大规模 (100鸡蛋, 1000层)"),
    ]

    results = []

    print(f"\n{'描述':<25}{'结果':<8}{'时间(ms)':<12}{'复杂度类':<15}")
    print("-" * 60)

    for eggs, floors, desc in test_configs:
        # 使用优化版本
        start = time.perf_counter()
        result = egg_drop_dp_optimized(eggs, floors)
        elapsed = (time.perf_counter() - start) * 1000

        # 判断复杂度类别
        if elapsed < 10:
            complexity = "极快"
        elif elapsed < 100:
            complexity = "快"
        elif elapsed < 1000:
            complexity = "中等"
        elif elapsed < 5000:
            complexity = "慢"
        else:
            complexity = "极慢"

        results.append({
            'eggs': eggs,
            'floors': floors,
            'result': result,
            'time_ms': elapsed,
            'desc': desc
        })

        print(f"{desc:<25}{result:<8}{elapsed:<12.2f}{complexity:<15}")

    return results


# ============================================================
# 最大规模测试
# ============================================================
def find_max_scale():
    """
    找出在有限时间内能处理的最大数据规模
    """
    print("\n" + "=" * 60)
    print("最大规模测试（时间限制：10秒）")
    print("=" * 60)

    time_limit = 10.0  # 秒
    max_results = []

    # 固定鸡蛋数，增加楼层数
    for eggs in [2, 3, 5, 10, 20, 50]:
        lo, hi = 1, 100000
        max_floors = 0
        max_time = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                start = time.perf_counter()
                result = egg_drop_dp_optimized(eggs, mid)
                elapsed = time.perf_counter() - start

                if elapsed < time_limit:
                    max_floors = mid
                    max_time = elapsed
                    lo = mid + 1
                else:
                    hi = mid - 1
            except (MemoryError, RecursionError):
                hi = mid - 1

        max_results.append({
            'eggs': eggs,
            'max_floors': max_floors,
            'time_ms': max_time * 1000
        })
        print(f"鸡蛋数={eggs:3d}: 最大楼层数={max_floors:6d}, 耗时={max_time*1000:.2f}ms")

    return max_results


# ============================================================
# 空间复杂度分析
# ============================================================
def space_analysis():
    """
    分析不同算法的空间复杂度
    """
    print("\n" + "=" * 60)
    print("空间复杂度分析")
    print("=" * 60)

    import tracemalloc

    test_cases = [
        (2, 100),
        (2, 500),
        (5, 100),
        (5, 500),
        (10, 500),
        (10, 1000),
    ]

    print(f"\n{'鸡蛋数':<8}{'楼层数':<8}{'空间(KB)':<12}{'理论空间(KB)':<15}")
    print("-" * 50)

    for eggs, floors in test_cases:
        tracemalloc.start()
        egg_drop_dp_optimized(eggs, floors)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 理论空间：dp表大小 (eggs+1) * (floors+1) * 8 bytes (int)
        theoretical = (eggs + 1) * (floors + 1) * 8 / 1024

        print(f"{eggs:<8}{floors:<8}{peak/1024:<12.2f}{theoretical:<15.2f}")


# ============================================================
# 理论复杂度分析
# ============================================================
def complexity_analysis():
    """
    理论复杂度分析
    """
    print("\n" + "=" * 60)
    print("理论复杂度分析")
    print("=" * 60)

    print("""
    1. 基础DP解法：
       - 时间复杂度：O(e * f^2)
       - 空间复杂度：O(e * f)
       - 其中 e 为鸡蛋数，f 为楼层数

    2. 二分查找优化DP：
       - 时间复杂度：O(e * f * log(f))
       - 空间复杂度：O(e * f)

    3. 交替DP状态定义：
       - 时间复杂度：O(e * answer)
       - 空间复杂度：O(e)
       - answer 为最终答案，通常远小于 f

    4. 记忆化搜索：
       - 时间复杂度：O(e * f^2)（最坏情况）
       - 空间复杂度：O(e * f)（递归栈 + 缓存）
    """)

    # 实际测量时间增长
    print("实际时间增长测试（固定eggs=2）：")
    print(f"{'楼层数':<10}{'时间(ms)':<12}{'时间比':<10}")
    print("-" * 35)

    prev_time = None
    for floors in [100, 200, 400, 800, 1600]:
        start = time.perf_counter()
        egg_drop_dp_optimized(2, floors)
        elapsed = (time.perf_counter() - start) * 1000

        ratio = elapsed / prev_time if prev_time else 1.0
        prev_time = elapsed

        print(f"{floors:<10}{elapsed:<12.2f}{ratio:<10.2f}")


# ============================================================
# 效率改进方案分析
# ============================================================
def efficiency_improvement_analysis():
    """
    分析算法效率改进空间
    """
    print("\n" + "=" * 60)
    print("效率改进方案分析")
    print("=" * 60)

    print("""
    1. 时间效率改进：

       a) 二分查找优化：
          - 原理：dp[i-1][x-1] 随 x 单调递增，dp[i][j-x] 随 x 单调递减
          - 可以用二分查找找到使两者最大值最小的 x
          - 时间复杂度从 O(e*f^2) 降低到 O(e*f*log(f))

       b) 交替DP状态定义：
          - 状态：dp[i][j] = i 次试验、j 个鸡蛋能确定的最大楼层数
          - 时间复杂度：O(e * answer)，answer 通常远小于 f
          - 这是最优解法

       c) 数学方法（凸优化）：
          - 利用函数的凸性，可以用三分查找代替二分查找
          - 进一步优化常数因子

    2. 空间效率改进：

       a) 滚动数组：
          - dp[i][] 只依赖于 dp[i-1][] 和 dp[i][]
          - 可以只保留两行，空间从 O(e*f) 降低到 O(f)

       b) 交替DP的空间优化：
          - 只需要一维数组 dp[eggs+1]
          - 空间复杂度：O(e)

    3. 实际改进效果：
    """)

    # 比较不同方法的性能
    eggs, floors = 10, 1000

    print(f"测试数据：{eggs} 个鸡蛋，{floors} 层楼")
    print(f"\n{'方法':<20}{'时间(ms)':<12}{'加速比':<10}")
    print("-" * 45)

    # 基础DP
    start = time.perf_counter()
    r1 = egg_drop_dp(eggs, floors)
    t1 = (time.perf_counter() - start) * 1000

    # 优化DP
    start = time.perf_counter()
    r2 = egg_drop_dp_optimized(eggs, floors)
    t2 = (time.perf_counter() - start) * 1000

    # 交替DP
    start = time.perf_counter()
    r3 = egg_drop_dp_alt(eggs, floors)
    t3 = (time.perf_counter() - start) * 1000

    print(f"{'基础DP':<20}{t1:<12.2f}{'1.00x':<10}")
    print(f"{'二分优化DP':<20}{t2:<12.2f}{t1/t2:<10.2f}x")
    print(f"{'交替DP':<20}{t3:<12.2f}{t1/t3:<10.2f}x")

    print(f"\nResult verification: {r1} = {r2} = {r3} PASS" if r1 == r2 == r3 else f"\nResult inconsistent!")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("鸡蛋掉落问题 - 动态规划实验")
    print("=" * 60)

    # 1. 正确性验证
    verify_correctness()

    # 2. 性能测试
    performance_test()

    # 3. 最大规模测试
    find_max_scale()

    # 4. 空间分析
    space_analysis()

    # 5. 理论复杂度分析
    complexity_analysis()

    # 6. 效率改进分析
    efficiency_improvement_analysis()

"""
边界情况分析 + 其他算法对比
"""

import time

from egg_drop import egg_drop_dp, egg_drop_dp_optimized, egg_drop_dp_alt, egg_drop_math


# ============================================================
# 边界情况分析
# ============================================================

def test_boundary_cases():
    """
    边界情况测试
    """
    print("=" * 70)
    print("边界情况分析")
    print("=" * 70)

    # 1. 极端鸡蛋数
    print("\n[1] 极端鸡蛋数")
    print("-" * 50)
    print(f"{'eggs':<8}{'floors':<10}{'answer':<10}{'time(ms)':<12}{'说明'}")
    print("-" * 50)

    cases = [
        (0, 100, "0个鸡蛋"),
        (1, 100, "1个鸡蛋（最少）"),
        (2, 100, "2个鸡蛋"),
        (100, 100, "鸡蛋数=楼层数"),
        (1000, 100, "鸡蛋数>>楼层数"),
        (10000, 100, "鸡蛋数>>>楼层数"),
    ]

    for eggs, floors, desc in cases:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        t = (time.perf_counter() - start) * 1000
        print(f"{eggs:<8}{floors:<10}{answer:<10}{t:<12.4f}{desc}")

    # 2. 极端楼层数
    print("\n[2] 极端楼层数")
    print("-" * 50)
    print(f"{'eggs':<8}{'floors':<12}{'answer':<10}{'time(ms)':<12}{'说明'}")
    print("-" * 50)

    cases = [
        (2, 0, "0层楼"),
        (2, 1, "1层楼"),
        (2, 2, "2层楼"),
        (2, 1000000, "百万层"),
        (2, 10000000, "千万层"),
        (10, 100000000, "亿层"),
    ]

    for eggs, floors, desc in cases:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        t = (time.perf_counter() - start) * 1000
        print(f"{eggs:<8}{floors:<12}{answer:<10}{t:<12.4f}{desc}")

    # 3. 特殊情况
    print("\n[3] 特殊情况")
    print("-" * 50)
    print(f"{'eggs':<8}{'floors':<10}{'answer':<10}{'说明'}")
    print("-" * 50)

    # 鸡蛋数 >= log2(floors) 时，等价于二分查找
    for floors in [10, 100, 1000, 10000]:
        eggs = floors  # 鸡蛋充足
        answer = egg_drop_dp_alt(eggs, floors)
        import math
        binary = math.ceil(math.log2(floors + 1))
        print(f"{eggs:<8}{floors:<10}{answer:<10}二分查找需要 {binary} 次")

    # 4. 边界值测试
    print("\n[4] 边界值测试")
    print("-" * 50)

    # 验证 dp[e][1] = 1
    print("dp[e][1] = 1 (任意鸡蛋数，1层楼):")
    for eggs in [1, 2, 5, 10, 100]:
        answer = egg_drop_dp_alt(eggs, 1)
        print(f"  dp[{eggs}][1] = {answer}")

    # 验证 dp[1][f] = f
    print("\ndp[1][f] = f (1个鸡蛋，任意楼层数):")
    for floors in [1, 5, 10, 20, 50]:
        answer = egg_drop_dp_alt(1, floors)
        print(f"  dp[1][{floors}] = {answer}")

    # 验证 dp[e][f] 的单调性
    print("\n单调性验证 (dp[e][f] 随 f 递增):")
    eggs = 3
    prev = 0
    monotone = True
    for floors in range(1, 51):
        answer = egg_drop_dp_alt(eggs, floors)
        if answer < prev:
            monotone = False
            print(f"  FAIL: dp[{eggs}][{floors}] = {answer} < dp[{eggs}][{floors-1}] = {prev}")
        prev = answer
    if monotone:
        print(f"  PASS: dp[{eggs}][1..50] 单调递增")

    print("\n单调性验证 (dp[e][f] 随 e 递减):")
    floors = 100
    prev = float('inf')
    monotone = True
    for eggs in range(1, 21):
        answer = egg_drop_dp_alt(eggs, floors)
        if answer > prev:
            monotone = False
            print(f"  FAIL: dp[{eggs}][{floors}] = {answer} > dp[{eggs-1}][{floors}] = {prev}")
        prev = answer
    if monotone:
        print(f"  PASS: dp[1..20][{floors}] 单调递减")


# ============================================================
# 其他算法对比
# ============================================================

def egg_drop_binary_search_greedy(eggs: int, floors: int) -> int:
    """
    朴素二分查找（不考虑鸡蛋限制）

    假设鸡蛋充足，直接用二分查找
    返回二分查找的次数（下限估计）
    """
    if floors <= 0:
        return 0
    import math
    return math.ceil(math.log2(floors + 1))


def egg_drop_greedy_adaptive(eggs: int, floors: int) -> int:
    """
    自适应贪心算法

    根据剩余鸡蛋数和楼层数动态调整策略：
    - 鸡蛋充足时用二分查找
    - 鸡蛋不足时用均匀分割
    """
    if eggs <= 0 or floors <= 0:
        return 0
    if eggs == 1:
        return floors

    # 使用DP的思想，但用贪心选择楼层
    # 这实际上就是交替DP的思路
    return egg_drop_dp_alt(eggs, floors)


def egg_drop_naive_binary(eggs: int, floors: int) -> int:
    """
    朴素二分查找（不考虑鸡蛋限制）

    假设鸡蛋充足，直接用二分查找
    返回二分查找的次数
    """
    if floors <= 0:
        return 0
    import math
    return math.ceil(math.log2(floors + 1))


def test_algorithm_comparison():
    """
    与其他算法的对比
    """
    print("\n" + "=" * 70)
    print("与其他算法的对比")
    print("=" * 70)

    # 1. 与朴素二分查找对比
    print("\n[1] 与朴素二分查找对比")
    print("-" * 60)
    print(f"{'eggs':<8}{'floors':<10}{'最优解':<10}{'二分查找':<10}{'差异':<10}{'说明'}")
    print("-" * 60)

    for floors in [10, 50, 100, 500, 1000]:
        for eggs in [1, 2, 5]:
            optimal = egg_drop_dp_alt(eggs, floors)
            binary = egg_drop_naive_binary(eggs, floors)
            diff = optimal - binary
            note = ""
            if eggs == 1:
                note = "1个鸡蛋，必须逐层"
            elif optimal == binary:
                note = "等于二分查找"
            elif optimal > binary:
                note = f"比二分多 {diff} 次"
            else:
                note = "比二分少（不可能）"
            print(f"{eggs:<8}{floors:<10}{optimal:<10}{binary:<10}{diff:<+10}{note}")

    # 2. 鸡蛋数对结果的影响
    print("\n[2] 鸡蛋数对结果的影响 (floors=1000)")
    print("-" * 60)
    print(f"{'eggs':<8}{'answer':<10}{'比值':<12}{'说明'}")
    print("-" * 60)

    floors = 1000
    prev_answer = None
    prev_eggs = None
    for eggs in [1, 2, 3, 5, 10, 20, 50, 100, 1000]:
        answer = egg_drop_dp_alt(eggs, floors)
        if prev_answer:
            ratio = prev_answer / answer
            note = f"比 {prev_eggs} 个鸡蛋快 {ratio:.2f}x"
        else:
            ratio = 0
            note = "基准"
        prev_answer = answer
        prev_eggs = eggs
        print(f"{eggs:<8}{answer:<10}{ratio if ratio else '-':<12}{note}")

    # 3. 不同算法的适用场景
    print("\n[3] 不同算法的适用场景")
    print("-" * 60)

    scenarios = [
        ("鸡蛋充足 (e >= log2(f))", "二分查找最优", 100, 100),
        ("鸡蛋很少 (e=1)", "必须逐层尝试", 1, 100),
        ("鸡蛋适中 (e=2)", "交替DP最优", 2, 100),
        ("鸡蛋较多 (e=10)", "交替DP最优", 10, 1000),
    ]

    for desc, note, eggs, floors in scenarios:
        optimal = egg_drop_dp_alt(eggs, floors)
        binary = egg_drop_naive_binary(eggs, floors)
        print(f"{desc:<25}最优解={optimal:<6}二分查找={binary:<6}{note}")


def test_greedy_vs_optimal():
    """
    贪心策略 vs 最优策略
    """
    print("\n" + "=" * 70)
    print("贪心策略 vs 最优策略")
    print("=" * 70)

    print("""
    贪心策略的局限性：

    贪心策略：每次选择"看起来最优"的楼层（如中间楼层）
    最优策略：考虑所有可能的选择，取全局最优

    示例：2个鸡蛋，10层楼

    贪心策略（二分查找）：
    - 第1次：第5层
    - 如果碎了：需要在1-4层逐个试，最多再试4次，共5次
    - 如果没碎：在6-10层继续二分
    - 最坏情况：5次

    最优策略（交替DP）：
    - 第1次：第4层（不是第5层！）
    - 如果碎了：在1-3层逐个试，最多再试3次，共4次
    - 如果没碎：在5-10层继续
    - 最坏情况：4次

    为什么第4层比第5层更优？
    - 因为要平衡"碎了"和"没碎"两种情况的最坏代价
    - 第4层：碎了→3次，没碎→需要在6层中找，也是3次
    - 第5层：碎了→4次，没碎→需要在5层中找，也是4次
    - 但第4层的总次数更少！
    """)

    # 实际验证
    print("实际验证：")
    print("-" * 50)

    # 2个鸡蛋，10层楼
    eggs, floors = 2, 10
    optimal = egg_drop_dp_alt(eggs, floors)
    print(f"eggs={eggs}, floors={floors}")
    print(f"  最优解：{optimal}")
    print(f"  二分查找：{egg_drop_naive_binary(eggs, floors)}")
    print(f"  差异：{egg_drop_naive_binary(eggs, floors) - optimal} 次")

    # 2个鸡蛋，100层楼
    eggs, floors = 2, 100
    optimal = egg_drop_dp_alt(eggs, floors)
    print(f"\neggs={eggs}, floors={floors}")
    print(f"  最优解：{optimal}")
    print(f"  二分查找：{egg_drop_naive_binary(eggs, floors)}")
    print(f"  差异：{egg_drop_naive_binary(eggs, floors) - optimal} 次")

    # 2个鸡蛋，1000层楼
    eggs, floors = 2, 1000
    optimal = egg_drop_dp_alt(eggs, floors)
    print(f"\neggs={eggs}, floors={floors}")
    print(f"  最优解：{optimal}")
    print(f"  二分查找：{egg_drop_naive_binary(eggs, floors)}")
    print(f"  差异：{egg_drop_naive_binary(eggs, floors) - optimal} 次")


if __name__ == '__main__':
    test_boundary_cases()
    test_algorithm_comparison()
    test_greedy_vs_optimal()

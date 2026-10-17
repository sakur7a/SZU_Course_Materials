"""
分析 f, e 与算法效率的关系
"""

import time
import math

from egg_drop import egg_drop_dp, egg_drop_dp_optimized, egg_drop_dp_alt


def analyze_relationship():
    """
    分析 f, e 与算法效率的关系
    """
    print("=" * 70)
    print("f, e 与算法效率的关系分析")
    print("=" * 70)

    # 1. 固定e，变化f
    print("\n[1] 固定鸡蛋数，变化楼层数")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<12}{'answer':<8}{'answer/f':<12}{'time(us)':<12}{'time/f':<12}")
    print("-" * 70)

    eggs = 10
    for floors in [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10]:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        elapsed = (time.perf_counter() - start) * 1e6  # microseconds

        ratio_af = answer / floors
        ratio_tf = elapsed / floors

        print(f"{eggs:<6}{floors:<12}{answer:<8}{ratio_af:<12.8f}{elapsed:<12.2f}{ratio_tf:<12.8f}")

    # 2. 固定f，变化e
    print("\n[2] 固定楼层数，变化鸡蛋数")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<12}{'answer':<8}{'time(us)':<12}{'time/e':<12}")
    print("-" * 70)

    floors = 10**9
    for eggs in [1, 2, 3, 5, 10, 20, 50, 100, 1000]:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        elapsed = (time.perf_counter() - start) * 1e6

        ratio_te = elapsed / eggs

        print(f"{eggs:<6}{floors:<12}{answer:<8}{elapsed:<12.2f}{ratio_te:<12.4f}")

    # 3. answer 与 f 的关系
    print("\n[3] answer 与 f 的关系（固定e=2）")
    print("-" * 70)
    print(f"{'floors':<15}{'answer':<10}{'log2(f)':<10}{'answer/log2(f)':<15}")
    print("-" * 70)

    eggs = 2
    for floors in [10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**12]:
        answer = egg_drop_dp_alt(eggs, floors)
        log_f = math.log2(floors)
        ratio = answer / log_f

        print(f"{floors:<15}{answer:<10}{log_f:<10.2f}{ratio:<15.2f}")

    # 4. answer 与 e 的关系
    print("\n[4] answer 与 e 的关系（固定f=10^9）")
    print("-" * 70)
    print(f"{'eggs':<10}{'answer':<10}{'log2(f)':<10}{'answer/log2(f)':<15}")
    print("-" * 70)

    floors = 10**9
    log_f = math.log2(floors)
    for eggs in [1, 2, 3, 5, 10, 20, 50, 100]:
        answer = egg_drop_dp_alt(eggs, floors)
        ratio = answer / log_f

        print(f"{eggs:<10}{answer:<10}{log_f:<10.2f}{ratio:<15.2f}")

    # 5. 时间复杂度验证
    print("\n[5] 时间复杂度验证（交替DP: O(e * answer)）")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<12}{'answer':<8}{'e*answer':<12}{'time(us)':<12}{'time/(e*ans)':<15}")
    print("-" * 70)

    test_cases = [
        (2, 10**6),
        (2, 10**9),
        (2, 10**12),
        (5, 10**6),
        (5, 10**9),
        (5, 10**12),
        (10, 10**6),
        (10, 10**9),
        (10, 10**12),
        (50, 10**9),
        (50, 10**12),
        (100, 10**9),
        (100, 10**12),
    ]

    for eggs, floors in test_cases:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        elapsed = (time.perf_counter() - start) * 1e6

        work = eggs * answer
        ratio = elapsed / work if work > 0 else 0

        print(f"{eggs:<6}{floors:<12}{answer:<8}{work:<12}{elapsed:<12.2f}{ratio:<15.4f}")


def explain_relationship():
    """
    解释关系
    """
    print("\n" + "=" * 70)
    print("关系总结")
    print("=" * 70)

    print("""
1. answer 与 f 的关系：
   - answer ≈ O(log f) （对数关系）
   - 当 e=2 时：answer ≈ 2 * sqrt(f)
   - 当 e>=log2(f) 时：answer ≈ log2(f)

2. answer 与 e 的关系：
   - e 越大，answer 越小
   - 当 e=1 时：answer = f （必须逐层）
   - 当 e>=log2(f) 时：answer = log2(f) （二分查找）
   - 当 1 < e < log2(f) 时：answer 介于两者之间

3. 时间复杂度：
   - 交替DP: O(e * answer)
   - 由于 answer = O(log f)，所以时间 = O(e * log f)
   - 但更精确地说：时间 = O(e * answer)，其中 answer << f

4. 关键洞察：
   - 算法效率主要取决于 answer，而不是 f
   - answer 是对数级别的，所以算法很快
   - e 增加会减少 answer，但增加 e 的系数
   - 存在最优的 e 使得总时间最小
    """)


if __name__ == '__main__':
    analyze_relationship()
    explain_relationship()

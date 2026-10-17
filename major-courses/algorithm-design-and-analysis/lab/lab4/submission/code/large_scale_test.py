"""
大规模数据测试 - 验证性能差异
"""

import time

from egg_drop import egg_drop_dp, egg_drop_dp_optimized, egg_drop_dp_alt, egg_drop_math


def test_why_so_fast():
    """
    分析为什么交替DP这么快
    """
    print("=" * 70)
    print("Why alt_DP is so fast?")
    print("=" * 70)

    eggs = 10
    floors_list = [1000, 5000, 10000, 50000, 100000]

    print(f"\neggs = {eggs}")
    print(f"\nfloors     answer      answer/floors    alt_DP(ms)")
    print("-" * 60)

    for floors in floors_list:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        t = (time.perf_counter() - start) * 1000
        ratio = answer / floors
        print(f"{floors:<10} {answer:<12} {ratio:<16.6f} {t:.4f}")

    answer_100k = egg_drop_dp_alt(eggs, 100000)
    print(f"""
Key insight:
- alt_DP complexity: O(eggs * answer)
- answer << floors
- For eggs={eggs}, floors=100000, answer={answer_100k}
- Actual work: {eggs} * {answer_100k} = {eggs * answer_100k}

- basic_DP complexity: O(eggs * floors^2)
- For eggs={eggs}, floors=100000: {eggs} * 100000^2 = {eggs * 100000 * 100000:.0e}

So basic_DP does ~{eggs * 100000 * 100000 / (eggs * answer_100k):.0e}x more work!
""")


def test_small_comparison():
    """
    小规模直接对比
    """
    print("=" * 70)
    print("Small scale comparison")
    print("=" * 70)

    test_cases = [
        (2, 100),
        (2, 200),
        (2, 500),
        (2, 1000),
        (3, 100),
        (3, 200),
        (3, 500),
        (5, 100),
        (5, 200),
        (5, 500),
        (10, 100),
        (10, 200),
        (10, 500),
    ]

    print(f"\neggs  floors   basic(ms)  bin(ms)    alt(ms)    speedup")
    print("-" * 65)

    for eggs, floors in test_cases:
        # basic DP
        start = time.perf_counter()
        r1 = egg_drop_dp(eggs, floors)
        t1 = (time.perf_counter() - start) * 1000

        # binary DP
        start = time.perf_counter()
        r2 = egg_drop_dp_optimized(eggs, floors)
        t2 = (time.perf_counter() - start) * 1000

        # alt DP
        start = time.perf_counter()
        r3 = egg_drop_dp_alt(eggs, floors)
        t3 = (time.perf_counter() - start) * 1000

        speedup = t1 / t3 if t3 > 0 else float('inf')

        print(f"{eggs:<5} {floors:<8} {t1:<10.2f} {t2:<10.2f} {t3:<10.4f} {speedup:.0f}x")


def test_medium_comparison():
    """
    中等规模对比（跳过基础DP）
    """
    print("\n" + "=" * 70)
    print("Medium scale comparison (skip basic DP)")
    print("=" * 70)

    test_cases = [
        (2, 2000),
        (2, 5000),
        (3, 2000),
        (3, 5000),
        (5, 2000),
        (5, 5000),
        (10, 2000),
        (10, 5000),
        (10, 10000),
        (20, 5000),
        (20, 10000),
        (50, 5000),
        (50, 10000),
    ]

    print(f"\neggs   floors    bin(ms)     alt(ms)     speedup(bin/alt)")
    print("-" * 60)

    for eggs, floors in test_cases:
        # binary DP
        start = time.perf_counter()
        r2 = egg_drop_dp_optimized(eggs, floors)
        t2 = (time.perf_counter() - start) * 1000

        # alt DP
        start = time.perf_counter()
        r3 = egg_drop_dp_alt(eggs, floors)
        t3 = (time.perf_counter() - start) * 1000

        speedup = t2 / t3 if t3 > 0 else float('inf')

        print(f"{eggs:<6} {floors:<9} {t2:<11.2f} {t3:<11.4f} {speedup:.0f}x")


def test_extreme():
    """
    极端测试 - 只用交替DP
    """
    print("\n" + "=" * 70)
    print("Extreme test (only alt_DP)")
    print("=" * 70)

    test_cases = [
        (2, 100000),
        (5, 100000),
        (10, 100000),
        (20, 100000),
        (50, 100000),
        (100, 100000),
        (10, 1000000),
        (50, 1000000),
        (100, 1000000),
        (10, 10000000),
        (50, 10000000),
        (100, 10000000),
    ]

    print(f"\neggs   floors       answer   time(ms)")
    print("-" * 50)

    for eggs, floors in test_cases:
        start = time.perf_counter()
        answer = egg_drop_dp_alt(eggs, floors)
        t = (time.perf_counter() - start) * 1000

        print(f"{eggs:<6} {floors:<12} {answer:<8} {t:.4f}")


def test_math_method():
    """
    测试组合数方法
    """
    print("\n" + "=" * 70)
    print("Math method (binomial coefficient)")
    print("=" * 70)

    test_cases = [
        (2, 100),
        (2, 1000),
        (2, 10000),
        (5, 1000),
        (5, 10000),
        (10, 1000),
        (10, 10000),
        (10, 100000),
        (50, 100000),
        (100, 100000),
        (10, 1000000),
        (50, 1000000),
        (100, 1000000),
    ]

    print(f"\neggs   floors      alt_DP     math       match?")
    print("-" * 55)

    for eggs, floors in test_cases:
        start = time.perf_counter()
        r1 = egg_drop_dp_alt(eggs, floors)
        t1 = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        r2 = egg_drop_math(eggs, floors)
        t2 = (time.perf_counter() - start) * 1000

        match = "YES" if r1 == r2 else "NO"
        print(f"{eggs:<6} {floors:<11} {t1:<10.4f} {t2:<10.4f} {match}")


if __name__ == '__main__':
    test_why_so_fast()
    test_small_comparison()
    test_medium_comparison()
    test_extreme()
    test_math_method()

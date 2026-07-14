"""
最大规模测试 - 快速版
"""

import time

from egg_drop import egg_drop_dp_optimized, egg_drop_dp_alt, egg_drop_math


def test_direct():
    """
    直接测试特定规模
    """
    print("=" * 70)
    print("最大规模测试（直接测试）")
    print("=" * 70)

    # 二分DP测试
    print("\n[1] 二分查找优化DP")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<12}{'result':<8}{'time(s)':<10}{'status'}")
    print("-" * 70)

    dp_tests = [
        (2, 10000),
        (2, 50000),
        (2, 100000),
        (2, 200000),
        (2, 500000),
        (5, 10000),
        (5, 50000),
        (5, 100000),
        (5, 200000),
        (10, 10000),
        (10, 50000),
        (10, 100000),
        (10, 200000),
        (20, 10000),
        (20, 50000),
        (20, 100000),
        (50, 10000),
        (50, 50000),
        (50, 100000),
        (100, 10000),
        (100, 50000),
        (100, 100000),
    ]

    for eggs, floors in dp_tests:
        try:
            start = time.perf_counter()
            result = egg_drop_dp_optimized(eggs, floors)
            elapsed = time.perf_counter() - start
            status = "OK" if elapsed < 10 else "SLOW"
            print(f"{eggs:<6}{floors:<12}{result:<8}{elapsed:<10.2f}{status}")
            if elapsed > 30:
                print("  -> 跳过更大规模")
                break
        except MemoryError:
            print(f"{eggs:<6}{floors:<12}{'N/A':<8}{'N/A':<10}OOM")
            break
        except Exception as e:
            print(f"{eggs:<6}{floors:<12}{'N/A':<8}{'N/A':<10}ERROR")
            break

    # 交替DP测试（可以处理更大的数据）
    print("\n[2] 交替DP（最优解法）")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<15}{'result':<8}{'time(s)':<10}{'status'}")
    print("-" * 70)

    alt_tests = [
        (2, 100000),
        (2, 1000000),
        (2, 10000000),
        (2, 100000000),
        (2, 1000000000),
        (5, 100000),
        (5, 1000000),
        (5, 10000000),
        (5, 100000000),
        (5, 1000000000),
        (10, 100000),
        (10, 1000000),
        (10, 10000000),
        (10, 100000000),
        (10, 1000000000),
        (20, 100000),
        (20, 1000000),
        (20, 10000000),
        (20, 100000000),
        (20, 1000000000),
        (50, 100000),
        (50, 1000000),
        (50, 10000000),
        (50, 100000000),
        (50, 1000000000),
        (100, 100000),
        (100, 1000000),
        (100, 10000000),
        (100, 100000000),
        (100, 1000000000),
    ]

    for eggs, floors in alt_tests:
        try:
            start = time.perf_counter()
            result = egg_drop_dp_alt(eggs, floors)
            elapsed = time.perf_counter() - start
            status = "OK" if elapsed < 10 else "SLOW"
            print(f"{eggs:<6}{floors:<15}{result:<8}{elapsed:<10.6f}{status}")
            if elapsed > 30:
                break
        except Exception as e:
            print(f"{eggs:<6}{floors:<15}{'N/A':<8}{'N/A':<10}ERROR")
            break

    # 组合数方法测试
    print("\n[3] 组合数方法")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<15}{'result':<8}{'time(s)':<10}{'status'}")
    print("-" * 70)

    math_tests = [
        (2, 100000),
        (2, 1000000),
        (2, 10000000),
        (5, 100000),
        (5, 1000000),
        (5, 10000000),
        (10, 100000),
        (10, 1000000),
        (10, 10000000),
        (10, 100000000),
        (50, 100000),
        (50, 1000000),
        (50, 10000000),
        (100, 100000),
        (100, 1000000),
        (100, 10000000),
    ]

    for eggs, floors in math_tests:
        try:
            start = time.perf_counter()
            result = egg_drop_math(eggs, floors)
            elapsed = time.perf_counter() - start
            status = "OK" if elapsed < 10 else "SLOW"
            print(f"{eggs:<6}{floors:<15}{result:<8}{elapsed:<10.6f}{status}")
            if elapsed > 30:
                break
        except Exception as e:
            print(f"{eggs:<6}{floors:<15}{'N/A':<8}{'N/A':<10}ERROR")
            break


if __name__ == '__main__':
    test_direct()

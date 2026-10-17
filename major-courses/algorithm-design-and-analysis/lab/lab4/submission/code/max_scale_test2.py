"""
超大规模测试 - 交替DP和组合数方法
"""

import time

from egg_drop import egg_drop_dp_alt, egg_drop_math


def test_ultra_large():
    """
    超大规模测试
    """
    print("=" * 70)
    print("超大规模测试")
    print("=" * 70)

    # 交替DP测试
    print("\n[1] 交替DP - 超大规模")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<20}{'result':<10}{'time(s)':<12}{'status'}")
    print("-" * 70)

    alt_tests = [
        # 2 eggs
        (2, 10**7),
        (2, 10**8),
        (2, 10**9),
        (2, 10**10),
        (2, 10**11),
        (2, 10**12),
        # 5 eggs
        (5, 10**7),
        (5, 10**8),
        (5, 10**9),
        (5, 10**10),
        (5, 10**11),
        (5, 10**12),
        # 10 eggs
        (10, 10**7),
        (10, 10**8),
        (10, 10**9),
        (10, 10**10),
        (10, 10**11),
        (10, 10**12),
        # 20 eggs
        (20, 10**9),
        (20, 10**10),
        (20, 10**11),
        (20, 10**12),
        # 50 eggs
        (50, 10**9),
        (50, 10**10),
        (50, 10**11),
        (50, 10**12),
        # 100 eggs
        (100, 10**9),
        (100, 10**10),
        (100, 10**11),
        (100, 10**12),
    ]

    for eggs, floors in alt_tests:
        try:
            start = time.perf_counter()
            result = egg_drop_dp_alt(eggs, floors)
            elapsed = time.perf_counter() - start
            status = "OK" if elapsed < 60 else "SLOW"
            print(f"{eggs:<6}{floors:<20}{result:<10}{elapsed:<12.6f}{status}")
            if elapsed > 30:
                print("  -> 太慢，跳过更大规模")
                break
        except Exception as e:
            print(f"{eggs:<6}{floors:<20}{'N/A':<10}{'N/A':<12}ERROR: {e}")
            break

    # 组合数方法测试
    print("\n[2] 组合数方法 - 超大规模")
    print("-" * 70)
    print(f"{'eggs':<6}{'floors':<20}{'result':<10}{'time(s)':<12}{'status'}")
    print("-" * 70)

    math_tests = [
        # 2 eggs
        (2, 10**7),
        (2, 10**8),
        (2, 10**9),
        (2, 10**10),
        (2, 10**11),
        (2, 10**12),
        # 5 eggs
        (5, 10**7),
        (5, 10**8),
        (5, 10**9),
        (5, 10**10),
        (5, 10**11),
        (5, 10**12),
        # 10 eggs
        (10, 10**7),
        (10, 10**8),
        (10, 10**9),
        (10, 10**10),
        (10, 10**11),
        (10, 10**12),
        # 20 eggs
        (20, 10**9),
        (20, 10**10),
        (20, 10**11),
        (20, 10**12),
        # 50 eggs
        (50, 10**9),
        (50, 10**10),
        (50, 10**11),
        (50, 10**12),
        # 100 eggs
        (100, 10**9),
        (100, 10**10),
        (100, 10**11),
        (100, 10**12),
    ]

    for eggs, floors in math_tests:
        try:
            start = time.perf_counter()
            result = egg_drop_math(eggs, floors)
            elapsed = time.perf_counter() - start
            status = "OK" if elapsed < 60 else "SLOW"
            print(f"{eggs:<6}{floors:<20}{result:<10}{elapsed:<12.6f}{status}")
            if elapsed > 30:
                print("  -> 太慢，跳过更大规模")
                break
        except Exception as e:
            print(f"{eggs:<6}{floors:<20}{'N/A':<10}{'N/A':<12}ERROR: {e}")
            break


if __name__ == '__main__':
    test_ultra_large()

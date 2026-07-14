# 推荐系统实验（C++）

## 实验目标
- 每个用户用 50 维兴趣向量表示。
- 对目标用户 `u`，计算与其他用户的余弦相似度。
- 基于全量排序取前 `k` 个最相似用户。
- 对比不同排序算法在 20 个随机样本下的平均运行时间。

## 计时口径与严谨性设置
- 计时仅包含推荐主流程：`相似度计算 + 全量排序 + 取前k`。
- 数据生成与文件读写不计入单次算法计时。
- 每个样本先做 2 次预热（warm-up），再计入 1 次有效测量。
- 每个 `(k, n, algorithm)` 使用 20 个随机样本，记录平均值、标准差、95% 置信区间。
- 使用 `n=100000` 作为基准点，按 `O(n log n)` 计算理论时间并与实测值对照。

## 对比算法
- `baseline_full_sort`：先计算全部相似度，再全量排序后取前 `k`。
- `optimized_topk_heap`：扫描过程中用大小为 `k` 的最小堆维护 Top-k，避免全量排序。

理论复杂度（`d=50`）：
- 基准算法：`O(nd + n log n + k)`
- 优化算法：`O(nd + n log k + k)`

## 运行步骤
1. 编译并运行 C++ 实验程序：

```powershell
g++ -O2 -std=c++17 code/dense/main.cpp -o bin/recommend_exp.exe
.\bin\recommend_exp.exe
```

2. 生成曲线图：

```powershell
python scripts/plot_results.py
```

## 目录结构
- `code/`：C++ 源码（`dense` / `sparse` / `advanced`）
- `scripts/`：绘图脚本
- `results/`：实验数据与图表（按实验类别分子目录）
- `reports/`：报告文档
- `bin/`：可执行文件
- `docs/`：作业模板与提交文档

## 输出文件
- `results/dense/timing_results.csv`：实验数据（`k,n,algorithm,avg_ms,stddev_ms,ci95_ms,theory_ms,ratio_to_baseline`）
- `results/dense/result_k_10.png`
- `results/dense/result_k_50.png`
- `results/dense/result_k_100.png`
- `results/dense/result_theory_k_10.png`
- `results/dense/result_theory_k_50.png`
- `results/dense/result_theory_k_100.png`

## 图像含义
- `result_k_*.png`：基准与优化算法的实测时间曲线（含 95%CI 误差棒）。
- `result_theory_k_*.png`：同一算法的实测时间与理论时间对照曲线（基准点 `n=100000`）。

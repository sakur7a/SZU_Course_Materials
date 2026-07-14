# 稀疏向量推荐实验（选做加分）

## 1. 问题描述
在稀疏兴趣建模中，用户兴趣向量不再是长度为 `d` 的稠密数组，而是仅包含 `s` 个非零项的稀疏向量（`s << d`）。

例子：`{电影:0.8, 游戏:0.5}`。

实验目标：
- 设计稀疏向量存储方式。
- 分析余弦相似度复杂度变化。
- 比较基准算法（全量排序）与优化算法（Top-k最小堆）在不同 `s` 下的效率变化。

## 2. 稀疏存储设计
采用有序稀疏列表存储：
- 每个用户存储为 `vector<(idx, val)>`。
- `idx` 为兴趣维度下标，`val` 为非零权重。
- 列表按 `idx` 升序，便于双指针交集计算点积。

优点：
- 空间从 `O(d)` 降为 `O(s)`。
- 点积可只访问非零项，避免大量零乘法。

## 3. 算法原理与复杂度
### 3.1 稀疏余弦相似度
对两个有序稀疏列表，用双指针法求交集点积：
- 若下标相同，累加乘积并双指针后移。
- 否则小下标指针后移。

单次点积复杂度：`O(s_u + s_i)`，均匀稀疏时近似 `O(s)`。

### 3.2 两种Top-k策略
- 基准算法 `baseline_full_sort`：
  先计算全部相似度，再全量排序，取前 `k`。
  复杂度：`O(ns + n log n + k)`。

- 优化算法 `optimized_topk_heap`：
  扫描时维护大小为 `k` 的最小堆。
  复杂度：`O(ns + n log k + k)`。

当 `k << n` 时，`n log k` 明显小于 `n log n`，理论上优化算法更快。

## 4. 核心伪代码（非源码）
### 4.1 稀疏余弦相似度（双指针）
```text
SparseCosine(A, B):
  i <- 0, j <- 0, dot <- 0
  while i < |A| and j < |B|:
    if A[i].idx == B[j].idx:
      dot <- dot + A[i].val * B[j].val
      i <- i + 1, j <- j + 1
    else if A[i].idx < B[j].idx:
      i <- i + 1
    else:
      j <- j + 1
  return dot / (norm(A) * norm(B))
```

### 4.2 基准算法（全量排序）
```text
BaselineFullSort(users, target, k):
  S <- empty
  for each user i != target:
    sim <- SparseCosine(users[target], users[i])
    append (sim, i) to S
  sort S by sim descending
  return first k elements of S
```

### 4.3 优化算法（Top-k最小堆）
```text
OptimizedTopKHeap(users, target, k):
  H <- empty min-heap with capacity k
  for each user i != target:
    sim <- SparseCosine(users[target], users[i])
    if size(H) < k:
      push (sim, i) into H
    else if sim > top(H).sim:
      pop(H)
      push (sim, i) into H
  return all elements in H
```

## 5. 实验设置
- 稀疏维度：`d = 10000`
- 用户规模：`n = 30000`
- Top-k：`k = 10`
- `s` 取值：`[5, 10, 20, 40, 80, 160, 320]`
- 每个配置：`12` 个随机样本，预热 `1` 次
- 指标：平均耗时（ms）、标准差、95%CI
- 基准点：`s=40`，用复杂度缩放得到理论时间曲线

输出文件：
- `sparse_timing_results.csv`
- `sparse_result_measured.png`
- `sparse_result_theory.png`
- `sparse_result_speedup.png`

## 6. 实测结果摘要
| s | baseline_ms | optimized_ms | speedup (baseline / optimized) |
|---|-------------|--------------|---------------------------------|
| 5 | 1.055 | 0.793 | 1.331x |
| 10 | 2.025 | 1.604 | 1.262x |
| 20 | 4.340 | 3.956 | 1.097x |
| 40 | 9.145 | 8.873 | 1.031x |
| 80 | 19.273 | 18.385 | 1.048x |
| 160 | 39.319 | 37.033 | 1.062x |
| 320 | 75.826 | 75.402 | 1.006x |

理论一致性（平均相对误差 MRE）：

| algorithm | MRE |
|-----------|-----|
| baseline_full_sort | 28.45% |
| optimized_topk_heap | 18.33% |

## 7. 分析与结论
1. 稀疏化显著降低了相似度计算的维度开销。
- 稠密场景单次相似度约为 `O(d)`，稀疏后约为 `O(s)`。
- 当 `s` 小时，推荐总耗时显著下降。

2. 不同 `s` 对效率影响明显。
- 两种算法耗时均随 `s` 近似线性上升，说明 `ns` 项逐渐主导。

3. 优化算法在小 `s` 下收益更明显。
- `s=5~10` 时加速比约 `1.26x~1.33x`。
- `s` 增大后，相似度计算项占主导，排序优化收益被稀释，`s=320` 时基本持平（约 `1.01x`）。

4. 理论与实测总体趋势一致，但非完全重合。
- 原因：常数项、缓存行为、内存分配、系统调度噪声，以及复杂度模型只描述主导项。

5. 对推荐系统工程实践的结论。
- 稀疏存储是大规模推荐的基础优化。
- 当 `k` 很小且 `s` 较小，Top-k最小堆比全量排序更高效。
- 当 `s` 较大时，优先优化相似度计算（并行化、SIMD、索引剪枝）更关键。

## 8. 进一步优化思考
1. 算法设计是否还能优化：可以。
- 当前流程仍需遍历全部用户，复杂度主项是 `O(ns)`；当 `n` 很大时，这一项仍是瓶颈。
- 可引入倒排索引（特征 -> 用户列表），查询时只访问目标用户非零特征相关候选，避免全库扫描。
- 可采用两阶段检索：先召回小候选集，再做精排 Top-k；若允许近似，可使用 ANN（如 HNSW）进一步降时延。

2. 代码层还能做哪些优化：
- 预计算避免重复计算：存储 `inv_norm = 1 / norm`，相似度计算由除法改为乘法链；复用中间缓冲减少频繁分配。
- 连续内存布局：将“每用户一个小vector”改为 CSR 风格（`indptr + indices + values`），提升顺序访问比例和 Cache 命中。
- 单精度浮点：将 `double` 改为 `float` 可降低内存带宽压力并提高向量化吞吐；需用 Top-k 重合率和排序稳定性做精度验证。

3. 预期收益与适用场景：
- 小 `s`、小 `k` 场景：Top-k最小堆 + 稀疏表示收益显著。
- 大 `s` 场景：相似度计算主导，优先做内存布局优化、并行化与向量化。
- 超大规模 `n` 场景：优先做候选召回（倒排/ANN），再做精排。

## 9. 倒排索引与两阶段检索实测
本节实现并测试了两项进一步优化：
- 倒排索引精确检索（`inverted_exact`）：建立“特征 -> 用户列表”，查询时仅访问目标用户非零特征相关用户。
- 两阶段检索（`two_stage_recall_rerank`）：先用受限倒排召回候选，再对候选做精确余弦精排 Top-k。

实验配置：
- `n=30000, d=10000, k=10`
- `s in [10, 20, 40, 80, 160, 320]`
- 每个 `s`：`8` 组样本，每组 `8` 次随机查询
- 两阶段参数：`posting_cap=200, candidate_cap=2000`

结果表：

| s | baseline_ms | inverted_ms | two_stage_ms | inverted speedup | two-stage speedup | two-stage recall@10 |
|---|-------------|-------------|--------------|------------------|-------------------|---------------------|
| 10 | 1.727 | 0.014 | 0.050 | 121.0x | 34.5x | 1.000 |
| 20 | 4.076 | 0.045 | 0.313 | 90.6x | 13.0x | 1.000 |
| 40 | 7.604 | 0.190 | 1.008 | 39.9x | 7.5x | 1.000 |
| 80 | 16.582 | 0.711 | 2.234 | 23.3x | 7.4x | 1.000 |
| 160 | 32.971 | 1.562 | 3.853 | 21.1x | 8.6x | 0.998 |
| 320 | 66.408 | 2.449 | 6.767 | 27.1x | 9.8x | 0.937 |

平均加速比：
- `inverted_exact` 相对全扫描基准：约 `53.8x`
- `two_stage_recall_rerank` 相对全扫描基准：约 `13.5x`

现象分析：
1. 倒排索引显著降低了查询延迟，说明“避免全库扫描”是最有效的算法级优化。
2. 两阶段检索在保持高召回的同时显著提速；`s` 增大时召回略有下降，体现了速度-精度权衡。
3. 索引构建本身有成本（`avg_index_build_ms` 随 `s` 增大而上升），适合离线构建并在线复用。

新增文件：
- `sparse_advanced_experiment.cpp`
- `sparse_advanced_results.csv`
- `plot_sparse_advanced.py`
- `sparse_advanced_latency.png`
- `sparse_advanced_candidates.png`
- `sparse_advanced_recall.png`

## 11. 统一实验：算法层与代码层优化叠加效果
为量化“代码层优化”和“算法层优化”的相对贡献，本节在同一批数据上同时比较四种方法：

1. `baseline_scan_aos_double`：AOS + double + 全扫描（基准）
2. `code_only_scan_csr_float`：仅代码层优化（CSR + float + inv_norm）
3. `algo_only_inverted_aos`：仅算法层优化（倒排精确检索）
4. `combined_inverted_csr_float`：算法层 + 代码层联合优化

实验配置：
- `n=20000, d=10000, k=10`
- `s in [20, 40, 80, 160, 320]`
- 每个 `s`：`6` 组样本，每组 `6` 次随机查询

结果文件：
- `results/advanced/unified_opt_results.csv`
- `results/advanced/unified_opt_latency.png`
- `results/advanced/unified_opt_speedup.png`
- `results/advanced/unified_opt_overlap.png`

延迟结果表（ms）：

| s | baseline | code-only | algo-only | combined |
|---|----------|-----------|-----------|----------|
| 20 | 2.818 | 2.295 | 0.036 | 0.029 |
| 40 | 5.933 | 4.899 | 0.120 | 0.099 |
| 80 | 12.242 | 10.103 | 0.488 | 0.424 |
| 160 | 24.303 | 19.970 | 1.119 | 0.950 |
| 320 | 48.754 | 40.119 | 1.729 | 1.388 |

平均加速比（相对 baseline）：
- `code_only_scan_csr_float`：约 `1.22x`（区间约 `1.20x~1.22x`）
- `algo_only_inverted_aos`：约 `40.46x`（区间约 `21.72x~77.98x`）
- `combined_inverted_csr_float`：约 `49.19x`（区间约 `25.57x~96.57x`）

Top-k 一致性（相对 baseline 的 overlap@10）：
- 三种优化方法平均 `overlap@10` 均为 `1.000`（本实验中未观察到排序结果差异）

结论：
1. 代码层优化能稳定降低常数开销，但提升量级有限（约 `1.2x`）。
2. 算法层优化（倒排避免全库扫描）带来数量级加速，是主要收益来源。
3. 代码层与算法层可以叠加，联合优化在所有 `s` 上均优于“仅算法层优化”。

## 10. 代码层优化实现与实测
本节落地实现了以下三项代码级优化，并进行了独立对比实验：
1. 预计算避免重复计算：为每个用户预存 `inv_norm = 1 / norm`，查询时将除法改为乘法链。
2. 连续内存布局：将用户稀疏向量从“每用户一个小 vector”改为 CSR（`indptr + indices + values`）。
3. 单精度浮点：将核心数据从 `double` 改为 `float`，并验证与基准结果的一致性（Top-k 重合率）。

实现文件：
- `code/advanced/sparse_lowlevel_opt_experiment.cpp`
- `scripts/plot_sparse_lowlevel_opt.py`

结果文件：
- `results/advanced/sparse_lowlevel_opt_results.csv`
- `results/advanced/sparse_lowlevel_opt_latency.png`
- `results/advanced/sparse_lowlevel_opt_overlap.png`

实验配置：
- `n=20000, d=10000, k=10`
- `s in [20, 40, 80, 160, 320]`
- 每个 `s`：`8` 组样本，每组 `8` 次随机查询

结果表：

| s | baseline_aos_double (ms) | optimized_csr_float_invnorm (ms) | speedup | overlap@10 |
|---|--------------------------|-----------------------------------|---------|------------|
| 20 | 2.993 | 2.448 | 1.223x | 1.000 |
| 40 | 5.894 | 4.921 | 1.198x | 1.000 |
| 80 | 13.884 | 11.700 | 1.187x | 1.000 |
| 160 | 26.207 | 21.710 | 1.207x | 1.000 |
| 320 | 49.909 | 41.380 | 1.206x | 1.000 |

平均加速比：约 `1.204x`。

结论：
1. 三项代码优化在所有 `s` 上都带来了稳定提速（约 `1.19x~1.22x`）。
2. `overlap@10 = 1.000`，说明在本实验规模下，`float` 替代 `double` 未引入可观测的 Top-k 质量损失。
3. 代码级优化主要降低常数开销；若追求数量级提速，仍需与倒排索引/两阶段召回等算法级优化结合。

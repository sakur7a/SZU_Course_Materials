# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

算法设计课程实验3：回溯法与地图填色问题。将地图区域抽象为无向图顶点着色问题，实现并比较多种着色算法。合法性统一用 `conflicts == 0` 判定。

## 运行方式

```bash
python run_experiments.py           # 主实验：小图 + DIMACS .col + 随机平面图 → outputs/summary.*
python run_baseline_compare.py      # 对比实验：暴力回溯 vs DSATUR → outputs/baseline_compare.csv
python make_figures.py              # 从 summary.csv 生成 matplotlib 图表 → outputs/figures/
python build_report.py              # 生成 docx + md 实验报告
python build_ppt_deck.py            # 生成 PPT 汇报幻灯片
```

## 依赖

- Python 3.12+，matplotlib，python-pptx，python-docx
- DIMACS 数据文件在 `地图数据/地图数据/*.col`

## 算法架构

核心实现在 `src/map_coloring.py`，提供统一接口 `Graph = list[set[int]]`（邻接表）和 `ColoringResult` 数据类。

**精确算法：**
- `brute_force_backtracking()` — 固点顺序暴力回溯，基线对比用
- `dsatur_coloring()` — 按饱和度选点 + 前向检查剪枝，小图/平面图用；支持 `find_all` 模式搜索所有解

**局部搜索：**
- `min_conflicts_coloring()` — 随机化 Min-Conflicts，贪心初始化 + 随机选冲突顶点
- `tabucol_coloring()` — TabuCol 禁忌表局部搜索，450 顶点 DIMACS 数据用；内置停滞重启机制

**图生成：**
- `small_image_graph()` — 实验用 9 顶点手工邻接图
- `random_apollonian_planar_graph(n, seed)` — 增量三角剖分生成随机平面图，保证平面性

## 实验流水线

1. `run_experiments.py` 调用 `src/map_coloring.py` 中的 `run_exact()`（DSATUR）和 `run_local()`（TabuCol）
2. 染色结果写入 `outputs/*.coloring.csv`，汇总写入 `summary.csv` / `summary.json`
3. `make_figures.py` 读取 summary 生成图表（运行时柱状图、平面图规模曲线、小图着色可视化）
4. `build_report.py` 和 `build_ppt_deck.py` 分别读取 summary + figures 生成最终文档

## 数据格式

- DIMACS .col：`p edge N M` 声明规模，`e u v` 声明边（1-indexed）
- 染色 CSV：`vertex,color`（1-indexed，-1 表示未着色）
- 所有脚本从 `outputs/summary.csv` 读取实验结果，先运行 `run_experiments.py` 再运行下游脚本

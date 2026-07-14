# 实验四：动态规划 — 鸡蛋掉落问题

## 目录结构

```
lab4/
├── src/                          # 源代码
│   ├── egg_drop.py              # 主要算法实现
│   ├── 实验报告.md               # 实验报告（Markdown）
│   └── tables_and_algorithms.tex # LaTeX表格和伪代码
│
├── pdf/                          # PDF文件
│   ├── 实验报告.pdf              # 完整实验报告
│   ├── 实验报告.tex              # LaTeX源文件
│   ├── tables_and_algorithms.pdf # 表格和伪代码PDF
│   └── tables_new.pdf           # 更新后的表格PDF
│
├── images/                       # 可视化图表
│   ├── performance_analysis.png  # 性能分析图
│   ├── dp_table_heatmap.png     # DP表热力图
│   ├── space_analysis.png       # 空间分析图
│   ├── algorithm_comparison.png # 算法对比图
│   ├── boundary_eggs.png        # 鸡蛋数边界分析
│   ├── boundary_floors.png      # 楼层数边界分析
│   ├── greedy_vs_optimal.png    # 贪心vs最优对比
│   └── summary_table.png        # 总结表
│
├── tests/                        # 测试脚本
│   ├── analysis.py              # 关系分析
│   ├── boundary_analysis.py     # 边界分析
│   ├── large_scale_test.py      # 大规模测试
│   ├── max_scale_test.py        # 最大规模测试
│   ├── max_scale_test2.py       # 超大规模测试
│   ├── visualization.py         # 可视化脚本
│   └── visualization2.py        # 可视化脚本2
│
└── templates/                    # 实验模板
    └── 实验4_动态规划（鸡蛋掉落问题）.doc
```

## 文件说明

### 主要文件

- `src/egg_drop.py`: 包含5种算法实现（基础DP、二分DP、交替DP、组合数、蛮力法）
- `src/实验报告.md`: 完整的实验报告，包含问题描述、算法分析、实验结果等
- `src/tables_and_algorithms.tex`: LaTeX格式的表格和伪代码

### 算法实现

1. **基础DP**: O(e×f²) 时间, O(e×f) 空间
2. **二分DP**: O(e×f×logf) 时间, O(e×f) 空间
3. **交替DP**: O(e×answer) 时间, O(e) 空间（最优）
4. **组合数**: O(answer×min(e,answer)) 时间, O(1) 空间
5. **蛮力法**: 仅用于验证正确性

### 关键结果

- 最大规模：交替DP可处理10¹²层楼（万亿级）
- 加速比：从261x到435,023x不等，取决于具体配置
- 数学本质：交替DP和组合数方法基于同一公式 ΣC(n,k)

## 使用方法

```bash
# 运行主要算法
python src/egg_drop.py

# 运行测试
python tests/large_scale_test.py

# 生成可视化
python tests/visualization.py

# 编译LaTeX
cd src && pdflatex tables_and_algorithms.tex
```

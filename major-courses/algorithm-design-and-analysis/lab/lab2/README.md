# 最近点对实验（蛮力法 + 分治法 + 可视化 + 报告）

## 目录说明
- `code/closest_pair.cpp`：核心算法与性能测试程序
- `code/run_experiment.ps1`：一键编译、运行、绘图、生成报告
- `viz/visualize_process.py`：算法过程动态可视化（支持读取分治轨迹CSV）
- `viz/plot_benchmark.py`：性能曲线绘制
- `report/generate_report.py`：根据CSV自动生成Markdown实验报告
- `results/`：运行后自动生成实验数据与图表

## 快速运行
在 PowerShell 中执行：

```powershell
cd code
./run_experiment.ps1
```

## 增强脚本用法（推荐）
- 快速小规模自测（更快出图）：

```powershell
cd code
./run_experiment.ps1 -Quick
```

- 自定义实验规模与重复次数：

```powershell
cd code
./run_experiment.ps1 -Repeats 5 -BenchmarkStart 100000 -BenchmarkEnd 1200000 -BenchmarkStep 100000 -BruteLimit 30000
```

- 常用开关：
	- `-SkipBenchmark`：跳过性能测试
	- `-SkipVisualization`：跳过绘图与过程可视化
	- `-SkipReport`：跳过报告生成
	- `-SkipExport`：跳过演示点导出
	- `-Python <解释器命令>`：指定 Python 命令（如 `python3`）

## 单独运行可视化
```powershell
python ../viz/visualize_process.py --points ../results/points_demo.csv --pair ../results/pair_demo.csv --trace ../results/trace_demo.csv
```

可视化新参数：
- `--view dashboard`：多面板仪表盘视图（全局点云 + 最优距离曲线 + 局部放大）
- `--view classic`：单图经典视图
- `--save ../results/process_demo.png`：保存高分辨率图片
- `--save-gif ../results/process_dashboard.gif`：导出过程动画 GIF
- `--no-show`：仅保存不弹窗（适合批处理）

过程轨迹文件说明：
- `results/trace_demo.csv` 由 C++ 分治算法在执行过程中导出，记录每次“当前最优距离被更新”的点对与距离。

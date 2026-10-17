# 最近点对实验 Oral PPT 叙事规划

## Audience
- 课程实验答辩老师与同学

## Objective
- 用约 5 分钟说明本实验中自己实现与优化的部分
- 强调分治算法工程化、过程可视化设计、实验结果分析
- 避免展开众所周知的基础定义与蛮力法细节

## Narrative Arc
1. 用一句话界定问题与实验目标
2. 交代采用的经典论文模型与本实验的工程实现范围
3. 说明自己设计的分治实现与可视化导出机制
4. 补充分治法合并步骤线性效率的几何证明
5. 展示算法执行过程图与 GIF
6. 展示性能结果与复杂度验证
7. 用结论页收束优势、适用场景与实验价值

## Slide List
1. 标题页：实验题目、核心结论、关键词
2. 论文模型与本实验定位：Shamos-Hoey 分治主线 + 本实验做了什么
3. 我实现的算法与优化：预排序、递归划分、带状区域、轨迹导出
4. 几何证明：2d×d 候选区、2×3 切分、最多 6 点比较
5. 可视化设计：GIF + dashboard 静态图 + 说明可视化表达的三层信息
6. 实验结果：可编辑图表 + 关键指标 + log-log 图
7. 结论：工程收益、复杂度验证、可复现性

## Source Plan
- `report/report_closest_pair.md`
- `results/benchmark.csv`
- `results/process_dashboard.png`
- `results/process_dashboard.gif`
- `results/benchmark_linear.png`
- `results/benchmark_loglog.png`

## Visual System
- 顶会 oral 风格
- 纯白背景
- 深灰正文，黑色标题
- 强调色仅使用蓝色与绿色
- 无阴影、无纹理、无拟物装饰

## Asset Needs
- 直接使用现有实验产物，不额外生成视觉素材
- 重点嵌入 `process_dashboard.gif`、`process_dashboard.png`

## Editability Plan
- 标题、正文、指标卡、结论均使用可编辑文本框
- 结果页使用原生图表对象呈现主要 benchmark 数据
- GIF/PNG 作为实验结果图像嵌入

# Face Recognition Reproducible Experiments

本仓库是计算机视觉实验三的人脸识别复现代码仓库，已包含实验所需数据集与核心代码。仓库目标是保证他人 clone 后可以在相同协议下复现实验结果，而不是保存报告、展示页面或中间图表。

## 已保留内容

- `face_recognition_system.py`：传统人脸识别核心实现。
- `scripts/`：传统方法、消融实验、鲁棒性实验和深度学习实验入口。
- `deep_learning/`：SmallCNN 与训练/评估工具。
- `data/`：FERET、Yale、Olivetti 三个实验数据集。
- `pyproject.toml` / `uv.lock`：传统实验 Python 依赖。
- `environment.yml`：深度学习实验 Conda 依赖。

未保留 GUI、Web Demo、报告生成、绘图导出、C++ 检测补充、MATLAB Gabor 补充和所有生成结果文件。实验运行后产生的 `results/`、`figures/`、`models/` 等目录会被 `.gitignore` 忽略。

## 仓库结构

```text
.
├── data/
│   ├── 数据库-feret_k175_s7_w80_h80/
│   │   └── feret_k175_s7_w80_h80/      # 1225 张 FERET BMP 图像
│   ├── yale_images.npy                 # Yale 图像数组
│   ├── yale_labels.npy                 # Yale 标签数组
│   ├── olivetti_images.npy             # Olivetti 图像数组
│   └── olivetti_labels.npy             # Olivetti 标签数组
├── deep_learning/
│   ├── cnn_model.py                    # SmallCNN / SmallCNNEmbedding
│   └── train.py                        # PyTorch 训练、评估与 k-shot 工具
├── scripts/
│   ├── prepare_datasets.py             # 数据集完整性检查
│   ├── run_quick_traditional_check.py  # 快速传统方法检查
│   ├── run_academic_experiments.py     # 7 组传统学术实验
│   ├── run_exp_12.py                   # 特征与分类器对比
│   ├── run_exp_34.py                   # 增强消融与过拟合分析
│   ├── run_exp_567.py                  # 统计、集成、跨数据集实验
│   ├── run_traditional_full_ablation.py
│   ├── run_final_candidates.py
│   ├── run_occlusion_block_voting.py
│   ├── run_dl_experiments.py           # 深度学习实验入口
│   ├── traditional_configs.py
│   └── results_io.py
├── face_recognition_system.py
├── pyproject.toml
├── uv.lock
└── environment.yml
```

## 环境配置

### 传统方法环境

推荐使用 `uv`：

```bash
uv sync
```

主要依赖：

| Package | Usage |
|---|---|
| numpy | 数组运算与数据存储 |
| opencv-python | 图像预处理、HOG、滤波与退化模拟 |
| scikit-learn | PCA、SVM、KNN、交叉验证与统计指标 |
| scipy | 置信区间与统计检验 |
| matplotlib | 可选结果可视化 |
| pillow | 图像读取与旋转增强 |
| insightface / onnxruntime | 预训练模型上限实验 |

### 深度学习环境

```bash
conda env create -f environment.yml
conda activate cv-deep
```

如果需要 GPU，请按本机 CUDA 版本安装匹配的 PyTorch。

## 数据集

数据已随仓库放在 `data/` 下。运行前可以执行完整性检查：

```bash
uv run python scripts/prepare_datasets.py --check
```

数据规格如下：

| Dataset | Subjects | Images | Size | Format |
|---|---:|---:|---:|---|
| FERET | 175 | 1225 | 80x80 | BMP, `{subject:03d}_{image:02d}.bmp` |
| Yale | 38 | 2414 | 80x80 | `yale_images.npy`, `yale_labels.npy` |
| Olivetti | 40 | 400 | 64x64 | `olivetti_images.npy`, `olivetti_labels.npy` |

说明：

- FERET 位于 `data/数据库-feret_k175_s7_w80_h80/feret_k175_s7_w80_h80/`。
- Yale 与 Olivetti 已预处理为 NumPy 数组。
- 部分脚本会在运行时将图像 resize 到统一输入尺寸。

## 快速复现

### 1. 快速检查传统流程

```bash
uv run python scripts/run_quick_traditional_check.py
```

该命令用于确认环境、数据路径和核心特征提取流程可运行。

### 2. 传统学术实验

完整 7 组实验：

```bash
uv run python scripts/run_academic_experiments.py
```

分组运行：

```bash
uv run python scripts/run_exp_12.py
uv run python scripts/run_exp_34.py
uv run python scripts/run_exp_567.py
```

完整传统消融：

```bash
uv run python scripts/run_traditional_full_ablation.py
```

最终候选配置与遮挡投票实验：

```bash
uv run python scripts/run_final_candidates.py
uv run python scripts/run_occlusion_block_voting.py
```

### 3. 深度学习实验

全部深度学习实验：

```bash
conda run -n cv-deep python scripts/run_dl_experiments.py
```

按实验类型运行：

```bash
conda run -n cv-deep python scripts/run_dl_experiments.py --experiment closedset
conda run -n cv-deep python scripts/run_dl_experiments.py --experiment kshot
conda run -n cv-deep python scripts/run_dl_experiments.py --experiment robustness
conda run -n cv-deep python scripts/run_dl_experiments.py --experiment pretrained
conda run -n cv-deep python scripts/run_dl_experiments.py --experiment cross
```

快速模式：

```bash
conda run -n cv-deep python scripts/run_dl_experiments.py --quick
```

## 方法概览

### 传统特征

| Method | Description |
|---|---|
| Eigenfaces | PCA with whitening，建模全局人脸结构 |
| LBP | 分块 Local Binary Pattern histogram，描述局部纹理 |
| Gabor | 多频率、多方向滤波器组，提取方向纹理响应 |
| HOG | Histogram of Oriented Gradients，统计梯度方向 |
| Combined | Eigenfaces + LBP，融合全局结构与局部纹理 |

### 分类器

| Classifier | Setting |
|---|---|
| SVM-RBF | `C=1`, `gamma=scale` |
| Linear SVM | `C=1` |
| KNN | `k=3`, cosine distance |
| Logistic Regression | `C=1` |
| Random Forest / GBDT | 分类器消融对照 |

### 深度模型

| Model | Description |
|---|---|
| SmallCNN | 轻量级卷积分类器 |
| SmallCNNEmbedding | 输出 128 维 embedding，用于最近邻或迁移评估 |

## 评估协议

| Dataset | Protocol | Notes |
|---|---|---|
| FERET | 7-fold stratified CV | 每折约每个身份 1 张测试图像 |
| Yale | 10-fold stratified CV | 保持身份比例一致 |
| Olivetti | 10-fold stratified CV | 小规模规整闭集基准 |
| Deep closed-set | 70/30 split | 训练集内部划分验证集用于 early stopping |

主要指标：

- Rank-1 Accuracy
- Rank-5 Accuracy
- Macro-F1
- 95% confidence interval
- Paired t-test

重要复现约束：

- 数据增强必须在每个训练折内部执行，不能先增强再划分。
- 默认随机种子见 `scripts/traditional_configs.py` 中的 `RANDOM_SEED = 42`。
- 深度学习结果可能受 PyTorch、CUDA、cuDNN 与 GPU 型号影响。

## 主要结果

### 传统方法最佳结果

| Dataset | Best Pipeline | Accuracy |
|---|---|---:|
| FERET | Combined(Eigenfaces+LBP) + SVM-RBF + hist_eq + augmentation | 93.5% |
| Yale | HOG + SVM-RBF | 98.2% |
| Olivetti | HOG + SVM-RBF + augmentation | 99.5% |

### 深度学习 closed-set 1:N

| Method | FERET | Yale | Olivetti |
|---|---:|---:|---:|
| PCA+SVM | 68.8% | 90.9% | 96.7% |
| LBP+SVM | 51.4% | 89.1% | 86.7% |
| HOG+SVM | 58.4% | 98.5% | 97.5% |
| SmallCNN | 79.3% | 78.8% | 97.5% |
| CNN+Aug | 87.2% | 95.1% | 94.2% |

### InsightFace 预训练上限

| Dataset | Linear SVM | KNN cosine |
|---|---:|---:|
| FERET | 92.7% | 88.6% |
| Yale | 97.4% | 95.9% |
| Olivetti | 100.0% | 100.0% |

## 输出文件

实验脚本会自动创建 `results/`，必要时也可能创建 `figures/`、`models/` 等目录。这些均为生成产物，不随仓库提交。

## License

MIT License.

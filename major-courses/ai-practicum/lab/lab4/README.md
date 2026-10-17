# 实验4

对实验3的 YOLO26m 检测教师进行 YOLO26n 特征蒸馏和 INT8 量化，并比较 PyTorch/TensorRT 部署。

无蒸馏学生 mAP50–95 为60.76%，蒸馏后61.43%；TensorRT FP16为61.46%/1.711ms，INT8为60.27%/1.175ms。

[Word 报告](实验4-完成报告.docx) · [PDF 报告](实验4-完成报告.pdf) · [核心源码](code/)

| 下载 | 内容 |
|---|---|
| [代码包](https://github.com/sakur7a/SZU_Course_Materials/releases/download/ai-practicum-lab3-lab4-20261018/lab4_code.zip) | 完整实验源码、配置、原始指标、SwanLab 本地记录、报告；不含图片和权重。 |
| [云端运行包](https://github.com/sakur7a/SZU_Course_Materials/releases/download/ai-practicum-lab3-lab4-20261018/lab4_cloud.zip) | 加入原始/派生数据、预训练与已训练权重，解压即可安装依赖后运行。 |

云端使用 Linux x86_64、单张 NVIDIA GPU，推荐 A100 40GB，驱动550.54.14或更新。首次依赖安装需要网络。

```bash
unzip lab4_cloud.zip
cd lab4_cloud
bash setup.sh
bash run.sh check
bash run.sh smoke
bash run.sh eval
# 重新训练：bash run.sh train
# 指定GPU：GPU_INDEX=1 bash run.sh eval
```

默认GPU0，启动时检查GPU是否空闲，并在tmux中运行。实验4云端包自带实验3选出的教师权重，TensorRT引擎会在目标GPU重新构建。所有新增环境、缓存和结果留在解压目录。

测试清单固定为87页、3,031个检测框；OCR为2,853个GT裁剪。检测AP不等于口算文字识别或批改正确率。延迟为A100上batch=1、960×960的GPU前向，预热30次后测量200次，不含预处理/传输/NMS。新云端结果以新run输出为准。

压缩包已通过解压、路径迁移、入口导入、绘图与GPU单图验证。验证复用了本机同版本环境，未在另一台云主机重新安装依赖。完整实验方法与限制见报告。

大文件通过 [Release](https://github.com/sakur7a/SZU_Course_Materials/releases/tag/ai-practicum-lab3-lab4-20261018) 分发，权重与数据不进入Git历史。

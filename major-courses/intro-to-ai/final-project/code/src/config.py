# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
网络配置文件：集中管理训练/评估阶段使用的超参数与运行配置。

本项目中，该配置会在 train.py 中被读取，用于：
- 数据输入尺寸、batch size 等数据相关设置
- 训练轮数、学习率、动量、权重衰减等优化器相关超参数
- checkpoint 的保存策略
- 运行平台（CPU/GPU/Ascend）与是否分布式训练

说明：
- 这里使用 easydict 的 EasyDict 让字典可以通过“点号”访问，例如 config.batch_size。
- set_config(args) 会把命令行参数 args 中的部分字段写入 config，保证训练脚本能统一读取。
"""
from easydict import EasyDict as ed

def set_config(args):
    """根据命令行参数 args 生成训练/评估需要的配置。

    Args:
        args: argparse 解析得到的参数对象，至少应包含：
            - platform: 运行平台字符串，例如 "CPU" / "GPU" / "Ascend"
            - run_distribute: 是否开启分布式训练（bool）

    Returns:
        EasyDict: 以属性方式访问的配置对象（例如 config.batch_size）。
    """
    if not args.run_distribute:
        args.run_distribute = False
    config_cpu = ed({
        # 分类类别数：猫/狗二分类，所以为 2
        "num_classes": 2,
        "image_height": 256,
        "image_width": 256,
        "batch_size": 32,
        "epoch_size": 30,
        # early stop 耐心值：当验证指标在连续 N 个 epoch 内未提升时提前停止训练
        # 0 表示关闭 early stop
        "early_stop_patience": 5,
        "warmup_epochs": 0, # warmup 轮数：学习率预热阶段的 epoch 数
        # 学习率初始值/结束值/峰值：用于 lr_generator.get_lr 生成每 step 的 lr
        # - lr_init: warmup 起点或初始学习率
        # - lr_max: 训练过程中的最大学习率
        # - lr_end: 训练末期的学习率
        "lr_init": .0,
        "lr_end": 0.01,
        "lr_max": 0.001,  
        "momentum": 0.9, # 动量系数：Momentum 优化器的超参数
        "weight_decay": 4e-5, # 权重衰减：L2 正则，防止过拟合
        "label_smooth": 0.1, # 标签平滑系数：>0 时会使用带 label smoothing 的交叉熵
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1, # 每隔多少个 epoch 保存一次 checkpoint
        "keep_checkpoint_max": 20, # 最多保留多少个 checkpoint 文件
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "run_distribute": args.run_distribute,
        "activation": "Softmax", # 激活函数

        # ---------- 类别映射 ----------
        "map":["cat", "dog"]
    })

    return config_cpu

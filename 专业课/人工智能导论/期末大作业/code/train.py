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
"""Train mobilenetV2 on ImageNet."""

import os
import time
import copy

from mindspore import Tensor, nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.common import set_seed

from src.dataset import extract_features
from src.lr_generator import get_lr
from src.config import set_config
from src.args import train_parse_args
from src.utils import context_device_init, export_mindir, predict_from_net, get_samples_from_eval_dataset
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt, get_networks, train

set_seed(1)


def _configure_matplotlib_cn():
    """配置 matplotlib 中文字体，避免图表中文乱码。"""
    import matplotlib

    try:
        from matplotlib import font_manager

        candidate_fonts = [
            "Microsoft YaHei",  # 微软雅黑（Windows 常见）
            "SimHei",           # 黑体
            "SimSun",           # 宋体
            "NSimSun",          # 新宋体
            "Arial Unicode MS", # 通用 Unicode 字体（部分系统有）
        ]
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in candidate_fonts:
            if font_name in available_fonts:
                matplotlib.rcParams["font.sans-serif"] = [font_name]
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _save_history_csv(history, out_dir, filename):
    csv_path = os.path.join(out_dir, filename)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,train_acc,eval_acc\n")
        losses = history.get("epoch_loss", [])
        train_acc = history.get("train_acc", [])
        eval_acc = history.get("eval_acc", [])
        for i, loss_val in enumerate(losses, start=1):
            ta = train_acc[i - 1] if i - 1 < len(train_acc) else ""
            ea = eval_acc[i - 1] if i - 1 < len(eval_acc) else ""
            f.write(f"{i},{loss_val},{ta},{ea}\n")
    return csv_path


def _plot_curve(x, y, title, xlabel, ylabel, out_path, label=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker='o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    if label is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _run_one_experiment(args_opt, base_config, test_list, *, tag, use_pretrain):
    """跑一次实验并返回 history。

    Args:
        tag: 输出目录与特征缓存标签，例如 "pretrain" / "scratch"。
        use_pretrain: True 表示加载 mobilenetV2.ckpt；False 表示随机初始化。
    """
    # 固定随机性，保证两次实验可比（尽量一致）
    set_seed(1)

    config = copy.deepcopy(base_config)
    base_out_dir = os.path.abspath(getattr(base_config, "save_checkpoint_path", "./"))
    out_dir = os.path.join(base_out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    config.save_checkpoint_path = out_dir

    # define network
    backbone_net, head_net, net = define_net(config, activation="Softmax")

    # 实验组：加载预训练权重；对照组：不加载（随机初始化）
    if use_pretrain:
        load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)
    else:
        # 不加载 ckpt，保持随机初始化；仍冻结 backbone（因为训练流程是“抽特征 + 训练 head”）
        for param in backbone_net.get_parameters():
            param.requires_grad = False

    # 训练前预测网格（保存，不弹窗）
    predict_from_net(
        net,
        test_list,
        config,
        show_title=f"{tag}: pre training",
        save_path=os.path.join(out_dir, f"predict_grid_{tag}_pre_training.png"),
        show=False,
    )

    # catch backbone features（不同 tag 写入不同 features 目录，避免混用）
    data, step_size = extract_features(backbone_net, args_opt.dataset_path, config, cache_tag=tag)

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # get learning rate
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size))

    # get optimizer
    opt = Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()), lr, config.momentum, config.weight_decay)

    # train
    train_net, eval_net = get_networks(head_net, loss, opt)
    history = train(train_net, eval_net, net, data, config)

    # 训练后预测网格（保存，不弹窗）
    predict_from_net(
        net,
        test_list,
        config,
        show_title=f"{tag}: after training",
        save_path=os.path.join(out_dir, f"predict_grid_{tag}_after_training.png"),
        show=False,
    )

    # 保存该实验的曲线与数据
    csv_path = _save_history_csv(history, out_dir, f"loss_history_{tag}.csv")
    print(f"[{tag}] history CSV: {csv_path}")

    losses = history.get("epoch_loss", [])
    if losses:
        _plot_curve(
            list(range(1, len(losses) + 1)),
            losses,
            title=f"{tag} 训练 Loss 下降趋势",
            xlabel="Epoch",
            ylabel="Loss",
            out_path=os.path.join(out_dir, f"loss_curve_{tag}.png"),
        )

    eval_acc = history.get("eval_acc", [])
    if eval_acc:
        _plot_curve(
            list(range(1, len(eval_acc) + 1)),
            eval_acc,
            title=f"{tag} 验证集准确率趋势",
            xlabel="Epoch",
            ylabel="Eval Acc",
            out_path=os.path.join(out_dir, f"eval_acc_curve_{tag}.png"),
        )

    return history, out_dir

if __name__ == '__main__':
    args_opt = train_parse_args()
    config = set_config(args_opt)
    start = time.time()

    # set context and device init
    context_device_init(config)

    # matplotlib 中文配置（用于后续对比曲线）
    try:
        _configure_matplotlib_cn()
    except Exception:
        pass

    # 从验证集中随机抽取：每类 10 张（猫 10 + 狗 10，共 20 张），便于统计准确率并标识错误样本
    test_list = get_samples_from_eval_dataset(args_opt.dataset_path, per_class=10)

    # -------- 对比实验：迁移学习 vs 随机初始化 --------
    # 实验组：加载 mobilenetV2.ckpt 预训练权重
    pretrain_history, pretrain_dir = _run_one_experiment(
        args_opt, config, test_list, tag="pretrain", use_pretrain=True
    )
    # 对照组：不加载 ckpt（随机初始化）
    scratch_history, scratch_dir = _run_one_experiment(
        args_opt, config, test_list, tag="scratch", use_pretrain=False
    )

    # -------- 画图比较（同一张图上画两条曲线）--------
    try:
        import matplotlib.pyplot as plt

        base_out_dir = os.path.abspath(getattr(config, "save_checkpoint_path", "./"))
        os.makedirs(base_out_dir, exist_ok=True)

        # 1) Loss 对比
        l1 = pretrain_history.get("epoch_loss", [])
        l2 = scratch_history.get("epoch_loss", [])
        if l1 or l2:
            plt.figure(figsize=(7, 4))
            if l1:
                plt.plot(range(1, len(l1) + 1), l1, marker='o', label='实验组：加载预训练权重')
            if l2:
                plt.plot(range(1, len(l2) + 1), l2, marker='o', label='对照组：随机初始化')
            plt.title("迁移学习对比：Loss 下降趋势")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(base_out_dir, "transfer_learning_compare_loss.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"对比图已保存: {out_path}")

        # 2) 验证集准确率对比
        a1 = pretrain_history.get("eval_acc", [])
        a2 = scratch_history.get("eval_acc", [])
        if a1 or a2:
            plt.figure(figsize=(7, 4))
            if a1:
                plt.plot(range(1, len(a1) + 1), a1, marker='o', label='实验组：加载预训练权重')
            if a2:
                plt.plot(range(1, len(a2) + 1), a2, marker='o', label='对照组：随机初始化')
            plt.title("迁移学习对比：验证集准确率趋势")
            plt.xlabel("Epoch")
            plt.ylabel("Eval Acc")
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(base_out_dir, "transfer_learning_compare_eval_acc.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"对比图已保存: {out_path}")
    except Exception as e:
        print(f"绘制对比曲线失败：{e}")

    print("train total cost {:5.4f} s".format(time.time() - start))

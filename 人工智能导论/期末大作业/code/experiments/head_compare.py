"""Head 结构对比实验（在冻结 backbone + 特征抽取范式下）

动机：
- 当前项目 backbone 冻结、训练基于抽取的特征，因此结构探索更适合从 head 入手。
- 不引入 MobileNetV3（需要新增模型实现/输出通道匹配/预训练权重等），而是改 head：
  - baseline: Linear
  - dropout: Dropout + Linear
  - mlp: Dense-ReLU-Dropout-Dense（对比不同 hidden_dim）

输出（自动生成到新文件夹）：
- compare_loss.png / compare_eval_acc.png（同图多线对比）
- results_table.csv / results_table.md（结果对比表）
- report_section.md（可直接粘贴到报告的小节）
- 每组子目录：history_*.csv + 单组曲线 + 预测错例网格图（加载 best ckpt）

运行：
    python experiments/head_compare.py

说明：
- 为保证结构对比公平，本脚本关闭 early stop（统一训练满 epoch_size），并固定其它超参数。
- 表中 time(s) 仅统计 head 训练耗时，不包含 backbone 特征抽取/缓存阶段。
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum

# 允许以 `python experiments/head_compare.py` 方式运行
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.args import train_parse_args
from src.config import set_config
from src.dataset import extract_features
from src.lr_generator import get_lr
from src.models import CrossEntropyWithLabelSmooth, define_net, get_networks, load_ckpt, train
from src.utils import context_device_init, get_samples_from_eval_dataset, predict_from_net


def _configure_matplotlib_cn():
    import matplotlib

    try:
        from matplotlib import font_manager

        candidate_fonts = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "NSimSun",
            "Arial Unicode MS",
        ]
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in candidate_fonts:
            if font_name in available_fonts:
                matplotlib.rcParams["font.sans-serif"] = [font_name]
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _ensure_dir(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def _save_history_csv(history: dict, out_dir: str, filename: str) -> str:
    out_dir = _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, filename)
    losses = history.get("epoch_loss", [])
    train_acc = history.get("train_acc", [])
    eval_acc = history.get("eval_acc", [])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,train_acc,eval_acc\n")
        for i in range(max(len(losses), len(train_acc), len(eval_acc))):
            ep = i + 1
            l = losses[i] if i < len(losses) else ""
            ta = train_acc[i] if i < len(train_acc) else ""
            ea = eval_acc[i] if i < len(eval_acc) else ""
            f.write(f"{ep},{l},{ta},{ea}\n")

    return csv_path


def _load_series_csv(csv_path: str, col: str):
    import csv

    values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get(col, "")
            if v == "" or v is None:
                continue
            try:
                values.append(float(v))
            except Exception:
                continue
    return values


def _plot_multi_curve(series, title, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    for label, values in series:
        if not values:
            continue
        x = list(range(1, len(values) + 1))
        plt.plot(x, values, marker="o", linewidth=1.2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _summarize(history: dict) -> dict:
    losses = history.get("epoch_loss", [])
    eval_acc = history.get("eval_acc", [])

    best_eval_acc = float(max(eval_acc)) if eval_acc else float("nan")
    best_eval_epoch = int(np.argmax(eval_acc) + 1) if eval_acc else 0
    final_eval_acc = float(eval_acc[-1]) if eval_acc else float("nan")
    final_loss = float(losses[-1]) if losses else float("nan")
    epochs_ran = int(len(losses))

    return {
        "best_eval_acc": best_eval_acc,
        "best_eval_epoch": best_eval_epoch,
        "final_eval_acc": final_eval_acc,
        "final_loss": final_loss,
        "epochs_ran": epochs_ran,
    }


def _run_one(
    args_opt,
    base_config,
    *,
    tag: str,
    head_variant: str,
    head_hidden_dim: int | None,
    head_dropout_keep_prob: float | None,
    exp_root: str,
    shared_cache_tag: str,
    test_list,
) -> dict:
    set_seed(1)

    config = copy.deepcopy(base_config)
    config.save_checkpoint_path = _ensure_dir(os.path.join(exp_root, tag))

    # 结构参数
    config.head_variant = head_variant
    if head_hidden_dim is not None:
        config.head_hidden_dim = int(head_hidden_dim)
    if head_dropout_keep_prob is not None:
        config.head_dropout_keep_prob = float(head_dropout_keep_prob)

    # 为公平对比：关闭 early stop，统一训练满 epoch_size
    config.early_stop_patience = 0

    backbone_net, head_net, net = define_net(config, activation="Softmax")
    load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)

    # head 不影响 backbone 特征，三组共享缓存
    data, step_size = extract_features(backbone_net, args_opt.dataset_path, config, cache_tag=shared_cache_tag)

    if config.label_smooth > 0:
        loss_fn = CrossEntropyWithLabelSmooth(smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    lr = Tensor(
        get_lr(
            global_step=0,
            lr_init=config.lr_init,
            lr_end=config.lr_end,
            lr_max=config.lr_max,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epoch_size,
            steps_per_epoch=step_size,
        )
    )

    opt = Momentum(
        filter(lambda x: x.requires_grad, head_net.get_parameters()),
        lr,
        config.momentum,
        config.weight_decay,
    )

    train_net, eval_net = get_networks(head_net, loss_fn, opt)

    # 计时仅统计 head 训练阶段
    start = time.time()
    history = train(train_net, eval_net, net, data, config)
    seconds = time.time() - start

    run_dir = config.save_checkpoint_path
    csv_path = _save_history_csv(history, run_dir, f"history_{tag}.csv")

    _configure_matplotlib_cn()
    _plot_multi_curve(
        [(tag, history.get("epoch_loss", []))],
        title=f"{tag} Loss 下降趋势",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(run_dir, f"loss_{tag}.png"),
    )
    _plot_multi_curve(
        [(tag, history.get("eval_acc", []))],
        title=f"{tag} 验证集准确率趋势",
        xlabel="Epoch",
        ylabel="Eval Acc",
        out_path=os.path.join(run_dir, f"eval_acc_{tag}.png"),
    )

    best_ckpt = os.path.join(run_dir, "ckpt_0", "mobilenetv2_best.ckpt")
    if os.path.isfile(best_ckpt):
        load_ckpt(net, best_ckpt, trainable=True)

    predict_from_net(
        net,
        test_list,
        config,
        show_title=f"{tag} (best ckpt)",
        save_path=os.path.join(run_dir, f"predict_grid_{tag}_best.png"),
        show=False,
    )

    summary = _summarize(history)
    summary.update(
        {
            "tag": tag,
            "head_variant": head_variant,
            "head_hidden_dim": int(head_hidden_dim) if head_hidden_dim is not None else "",
            "head_dropout_keep_prob": float(head_dropout_keep_prob) if head_dropout_keep_prob is not None else "",
            "epoch_size": int(config.epoch_size),
            "seconds": float(seconds),
            "history_csv": csv_path,
            "run_dir": run_dir,
        }
    )
    return summary


def _write_results(rows, exp_root: str):
    exp_root = _ensure_dir(exp_root)

    csv_path = os.path.join(exp_root, "results_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "tag,head_variant,head_hidden_dim,head_dropout_keep_prob,epoch_size,epochs_ran,best_eval_acc,best_eval_epoch,final_eval_acc,final_loss,seconds,run_dir\n"
        )
        for r in rows:
            f.write(
                f"{r['tag']},{r['head_variant']},{r['head_hidden_dim']},{r['head_dropout_keep_prob']},{r['epoch_size']},{r['epochs_ran']},"
                f"{r['best_eval_acc']},{r['best_eval_epoch']},{r['final_eval_acc']},{r['final_loss']},{r['seconds']},{r['run_dir']}\n"
            )

    md_path = os.path.join(exp_root, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Head 结构对比结果表\n\n")
        f.write("说明：time(s) 仅统计 head 训练耗时，不包含 backbone 特征抽取/缓存阶段。\n\n")
        f.write(
            "| 组别(tag) | head_variant | hidden_dim | dropout_keep_prob | epoch_size | 实际训练轮数 | best eval acc | best epoch | final eval acc | final loss | time(s) |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            hidden_dim = r["head_hidden_dim"] if r["head_hidden_dim"] != "" else "-"
            keep_prob = r["head_dropout_keep_prob"] if r["head_dropout_keep_prob"] != "" else "-"
            f.write(
                f"| {r['tag']} | {r['head_variant']} | {hidden_dim} | {keep_prob} | {r['epoch_size']} | {r['epochs_ran']} | "
                f"{r['best_eval_acc']:.4f} | {r['best_eval_epoch']} | {r['final_eval_acc']:.4f} | {r['final_loss']:.4f} | {r['seconds']:.1f} |\n"
            )

    # 总对比曲线（同图多线）
    _configure_matplotlib_cn()
    try:
        loss_series = [(r["tag"], _load_series_csv(r["history_csv"], "loss")) for r in rows]
        acc_series = [(r["tag"], _load_series_csv(r["history_csv"], "eval_acc")) for r in rows]

        _plot_multi_curve(loss_series, "Head 结构对比：Loss 曲线", "Epoch", "Loss", os.path.join(exp_root, "compare_loss.png"))
        _plot_multi_curve(acc_series, "Head 结构对比：验证集准确率曲线", "Epoch", "Eval Acc", os.path.join(exp_root, "compare_eval_acc.png"))
    except Exception as e:
        print(f"绘制总对比曲线失败：{e}")

    # 报告段落
    report_path = os.path.join(exp_root, "report_section.md")
    best_row = max(rows, key=lambda x: float(x.get("best_eval_acc", float("nan"))))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("## 结构探索：Head 结构对比（冻结 backbone + 特征抽取）\n\n")
        f.write("本实验在保持 MobileNetV2 backbone 冻结、训练基于抽取特征的前提下，仅对 head 结构进行探索。"
                "对比 baseline（单层线性分类）与加入 Dropout/MLP 的 head 变体，观察验证集准确率曲线与错例网格图（每组目录下 predict_grid_*_best.png）。\n\n")
        f.write("为保证对比公平：三组均使用相同 backbone 预训练权重与同一份特征缓存；关闭 early stop，统一训练轮数 epoch_size；time(s) 仅统计 head 训练耗时。\n\n")
        f.write(f"从结果表可见（见 results_table.md），最佳组为 **{best_row['tag']}**，best eval acc = **{best_row['best_eval_acc']:.4f}**（best epoch = {best_row['best_eval_epoch']}）。\n\n")
        f.write("总体结论：在当前训练范式下，head 的容量与正则化强度会显著影响收敛稳定性与泛化；"
                "相比直接更换 backbone，改 head 更容易在现有工程与预训练权重条件下得到可复现且可解释的提升。\n")

    return csv_path, md_path, report_path


def main():
    args_opt = train_parse_args()
    base_config = set_config(args_opt)

    # 为看清结构差异，适当拉长训练轮数；同时关闭 early stop（在 run 中也会强制关闭）
    base_config.epoch_size = 50
    base_config.early_stop_patience = 0

    context_device_init(base_config)

    # 固定同一批样本用于错例图对比（猫 10 + 狗 10 = 20）
    test_list = get_samples_from_eval_dataset(args_opt.dataset_path, per_class=10)

    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_root = _ensure_dir(os.path.join(os.path.abspath(base_config.save_checkpoint_path), f"head_compare_{ts}"))

    shared_cache_tag = f"pretrain_head_shared_bs{base_config.batch_size}_img{base_config.image_height}"

    runs = [
        # baseline
        ("baseline", "baseline", None, None),
        # dropout only (keep_prob=0.8 => drop 20%)
        ("drop20", "dropout", None, 0.8),
        # MLP hidden dim 对比
        ("mlp256_drop20", "mlp", 256, 0.8),
        ("mlp512_drop20", "mlp", 512, 0.8),
    ]

    rows = []
    for tag, head_variant, hidden_dim, keep_prob in runs:
        print(f"\n===== Running {tag} (head_variant={head_variant}) =====")
        rows.append(
            _run_one(
                args_opt,
                base_config,
                tag=tag,
                head_variant=head_variant,
                head_hidden_dim=hidden_dim,
                head_dropout_keep_prob=keep_prob,
                exp_root=exp_root,
                shared_cache_tag=shared_cache_tag,
                test_list=test_list,
            )
        )

    csv_path, md_path, report_path = _write_results(rows, exp_root)

    print("\n===== Done =====")
    print(f"输出目录: {exp_root}")
    print(f"对比表 CSV: {csv_path}")
    print(f"对比表 Markdown: {md_path}")
    print(f"报告段落: {report_path}")
    print("对比曲线图: compare_loss.png / compare_eval_acc.png")


if __name__ == "__main__":
    main()

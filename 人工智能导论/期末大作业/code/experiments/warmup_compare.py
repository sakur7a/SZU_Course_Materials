"""warmup_epochs 对比实验：0 vs 5 vs 10

目标：对比不同 warmup 轮数对收敛速度/稳定性与最终精度的影响。

- 固定使用迁移学习（加载 mobilenetV2.ckpt 预训练权重，冻结 backbone）
- 固定 batch_size / lr / label_smooth 等其它超参数
- 仅对比 warmup_epochs：
  - 0（不使用 warmup）
  - 5
  - 10

输出（自动生成到新文件夹）：
- compare_loss.png / compare_eval_acc.png（同图多线对比）
- results_table.csv / results_table.md（结果对比表）
- 每组子目录：history_*.csv + 单组曲线

运行：
    python experiments/warmup_compare.py

说明：
- 本项目训练方式为“先抽 backbone 特征、再训练 head”；warmup 只影响 head 的学习率曲线。
- 表中 time(s) 仅统计 head 训练耗时，不包含特征抽取/缓存阶段。
"""

import copy
import os
import sys
import time

import numpy as np
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum

# 允许以 `python experiments/warmup_compare.py` 方式运行
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.args import train_parse_args
from src.config import set_config
from src.dataset import extract_features
from src.lr_generator import get_lr
from src.models import CrossEntropyWithLabelSmooth, define_net, get_networks, load_ckpt, train
from src.utils import context_device_init


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


def _run_one(args_opt, base_config, *, warmup_epochs: int, tag: str, exp_root: str, shared_cache_tag: str) -> dict:
    set_seed(1)

    config = copy.deepcopy(base_config)
    config.warmup_epochs = int(warmup_epochs)
    # warmup 对比实验中关闭 early stop，避免其导致不同组训练轮数不同而干扰结论
    config.early_stop_patience = 0

    run_dir = _ensure_dir(os.path.join(exp_root, tag))
    config.save_checkpoint_path = run_dir

    backbone_net, head_net, net = define_net(config, activation="Softmax")
    load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)

    # warmup 不影响 backbone 特征，因此共享缓存避免重复计算
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

    summary = _summarize(history)
    summary.update(
        {
            "tag": tag,
            "warmup_epochs": int(warmup_epochs),
            "epoch_size": int(config.epoch_size),
            "seconds": float(seconds),
            "history_csv": csv_path,
            "run_dir": run_dir,
        }
    )
    return summary


def _write_results_table(rows, exp_root: str):
    exp_root = _ensure_dir(exp_root)

    csv_path = os.path.join(exp_root, "results_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "tag,warmup_epochs,epoch_size,epochs_ran,best_eval_acc,best_eval_epoch,final_eval_acc,final_loss,seconds,run_dir\n"
        )
        for r in rows:
            f.write(
                f"{r['tag']},{r['warmup_epochs']},{r['epoch_size']},{r['epochs_ran']},"
                f"{r['best_eval_acc']},{r['best_eval_epoch']},{r['final_eval_acc']},{r['final_loss']},{r['seconds']},"
                f"{r['run_dir']}\n"
            )

    md_path = os.path.join(exp_root, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# warmup_epochs 对比结果表\n\n")
        f.write("说明：time(s) 仅统计 head 训练耗时，不包含 backbone 特征抽取/缓存阶段。\n\n")
        f.write(
            "| 组别(tag) | warmup_epochs | epoch_size | 实际训练轮数 | best eval acc | best epoch | final eval acc | final loss | time(s) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['tag']} | {r['warmup_epochs']} | {r['epoch_size']} | {r['epochs_ran']} | "
                f"{r['best_eval_acc']:.4f} | {r['best_eval_epoch']} | {r['final_eval_acc']:.4f} | {r['final_loss']:.4f} | {r['seconds']:.1f} |\n"
            )

    return csv_path, md_path


def main():
    args_opt = train_parse_args()
    base_config = set_config(args_opt)

    # 拉长总轮数以便观察 warmup 对前期收敛的影响
    base_config.epoch_size = 50

    # 对比 warmup 时关闭 early stop，确保各组训练轮数一致
    base_config.early_stop_patience = 0

    context_device_init(base_config)

    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_root = _ensure_dir(
        os.path.join(os.path.abspath(base_config.save_checkpoint_path), f"warmup_compare_{ts}")
    )

    shared_cache_tag = f"pretrain_shared_bs{base_config.batch_size}_img{base_config.image_height}"

    runs = [
        (0, "wu0"),
        (5, "wu5"),
        (10, "wu10"),
    ]

    rows = []
    for warmup_epochs, tag in runs:
        print(f"\n===== Running {tag} (warmup_epochs={warmup_epochs}) =====")
        rows.append(
            _run_one(
                args_opt,
                base_config,
                warmup_epochs=warmup_epochs,
                tag=tag,
                exp_root=exp_root,
                shared_cache_tag=shared_cache_tag,
            )
        )

    _configure_matplotlib_cn()
    try:
        loss_series = [(r["tag"], _load_series_csv(r["history_csv"], "loss")) for r in rows]
        acc_series = [(r["tag"], _load_series_csv(r["history_csv"], "eval_acc")) for r in rows]

        _plot_multi_curve(loss_series, "warmup_epochs 对比：Loss 曲线", "Epoch", "Loss", os.path.join(exp_root, "compare_loss.png"))
        _plot_multi_curve(acc_series, "warmup_epochs 对比：验证集准确率曲线", "Epoch", "Eval Acc", os.path.join(exp_root, "compare_eval_acc.png"))
    except Exception as e:
        print(f"绘制总对比曲线失败：{e}")

    csv_path, md_path = _write_results_table(rows, exp_root)

    print("\n===== Done =====")
    print(f"输出目录: {exp_root}")
    print(f"对比表 CSV: {csv_path}")
    print(f"对比表 Markdown: {md_path}")
    print("对比曲线图: compare_loss.png / compare_eval_acc.png")


if __name__ == "__main__":
    main()

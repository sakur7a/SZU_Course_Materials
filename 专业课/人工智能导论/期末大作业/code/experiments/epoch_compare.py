"""参数对比实验稳、最容易写结论）：

- 固定使用迁移学习（加载 mobilenetV2.ckpt 预训练权重，冻结 backbone）
- 仅对比训练超参数：epoch_size（10/30/50）
- 输出：
  - 每组独立曲线与 CSV
  - 总对比曲线（同图多线）
  - 结果对比表（CSV + Markdown）

运行示例（在 code 目录下）：
    python experiments/param_compare.py --dataset_path dataset\PetImages --pretrain_ckpt ..\mobilenetV2.ckpt

说明：
- 本项目训练方式为“先抽取 backbone 特征、再训练 head”。
- batch_size 影响 feature 缓存的 shape，因此缓存目录需要按实验区分。
"""

import argparse
import copy
import os
import sys
import time

# 允许以 `python experiments/param_compare.py` 方式运行：
# 把 code/ 目录加入 sys.path，确保能导入 src/* 模块。
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum

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


def _plot_multi_curve(series, title, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    for label, values in series:
        if not values:
            continue
        x = list(range(1, len(values) + 1))
        plt.plot(x, values, marker='o', linewidth=1.2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
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

    return {
        "best_eval_acc": best_eval_acc,
        "best_eval_epoch": best_eval_epoch,
        "final_eval_acc": final_eval_acc,
        "final_loss": final_loss,
    }


def _run_one(args_opt, base_config, *, epoch_size: int, tag: str, exp_root: str) -> dict:
    # 为了可比性，固定随机种子
    set_seed(1)

    config = copy.deepcopy(base_config)
    config.epoch_size = int(epoch_size)

    run_dir = _ensure_dir(os.path.join(exp_root, tag))
    config.save_checkpoint_path = run_dir

    start = time.time()

    backbone_net, head_net, net = define_net(config, activation="Softmax")
    load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)

    # 特征缓存需要区分 epoch_size（虽然理论上特征和 epoch_size 无关，但避免误用老缓存更稳）
    cache_tag = f"pretrain_ep{config.epoch_size}_bs{config.batch_size}_wu{config.warmup_epochs}"
    data, step_size = extract_features(backbone_net, args_opt.dataset_path, config, cache_tag=cache_tag)

    if config.label_smooth > 0:
        loss_fn = CrossEntropyWithLabelSmooth(smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    lr = Tensor(get_lr(
        global_step=0,
        lr_init=config.lr_init,
        lr_end=config.lr_end,
        lr_max=config.lr_max,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epoch_size,
        steps_per_epoch=step_size,
    ))

    opt = Momentum(
        filter(lambda x: x.requires_grad, head_net.get_parameters()),
        lr,
        config.momentum,
        config.weight_decay,
    )

    train_net, eval_net = get_networks(head_net, loss_fn, opt)
    history = train(train_net, eval_net, net, data, config)

    seconds = time.time() - start

    csv_path = _save_history_csv(history, run_dir, f"history_{tag}.csv")

    # 单组曲线
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
    summary.update({
        "tag": tag,
        "epoch_size": int(config.epoch_size),
        "batch_size": int(config.batch_size),
        "warmup_epochs": int(config.warmup_epochs),
        "seconds": float(seconds),
        "history_csv": csv_path,
        "run_dir": run_dir,
    })
    return summary


def _write_results_table(rows, exp_root: str):
    exp_root = _ensure_dir(exp_root)

    # CSV
    csv_path = os.path.join(exp_root, "results_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tag,epoch_size,batch_size,warmup_epochs,best_eval_acc,best_eval_epoch,final_eval_acc,final_loss,seconds,run_dir\n")
        for r in rows:
            f.write(
                f"{r['tag']},{r['epoch_size']},{r['batch_size']},{r['warmup_epochs']},"
                f"{r['best_eval_acc']},{r['best_eval_epoch']},{r['final_eval_acc']},{r['final_loss']},{r['seconds']},"
                f"{r['run_dir']}\n"
            )

    # Markdown
    md_path = os.path.join(exp_root, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 参数对比结果表（迁移学习固定）\n\n")
        f.write("| 组别(tag) | epoch_size | batch_size | warmup_epochs | best eval acc | best epoch | final eval acc | final loss | time(s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['tag']} | {r['epoch_size']} | {r['batch_size']} | {r['warmup_epochs']} | "
                f"{r['best_eval_acc']:.4f} | {r['best_eval_epoch']} | {r['final_eval_acc']:.4f} | {r['final_loss']:.4f} | {r['seconds']:.1f} |\n"
            )

    return csv_path, md_path


def main():
    # 复用原项目参数解析（保持 platform/dataset_path/pretrain_ckpt 等一致）
    args_opt = train_parse_args()
    base_config = set_config(args_opt)

    # 初始化运行环境
    context_device_init(base_config)

    # 新建一个实验输出文件夹
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_root = _ensure_dir(os.path.join(os.path.abspath(base_config.save_checkpoint_path), f"param_compare_{ts}"))

    # 仅做“epoch_size 对比”三组（baseline=30，另加 10/50）
    epoch_list = [10, 30, 50]

    rows = []
    for ep in epoch_list:
        tag = f"ep{ep}"
        print(f"\n===== Running {tag} (epoch_size={ep}) =====")
        rows.append(_run_one(args_opt, base_config, epoch_size=ep, tag=tag, exp_root=exp_root))

    # 总对比曲线
    _configure_matplotlib_cn()
    try:
        loss_series = [(r["tag"], _load_series_csv(r["history_csv"], "loss")) for r in rows]
        acc_series = [(r["tag"], _load_series_csv(r["history_csv"], "eval_acc")) for r in rows]

        _plot_multi_curve(loss_series, "epoch_size 对比：Loss 曲线", "Epoch", "Loss",
                          os.path.join(exp_root, "compare_loss.png"))
        _plot_multi_curve(acc_series, "epoch_size 对比：验证集准确率曲线", "Epoch", "Eval Acc",
                          os.path.join(exp_root, "compare_eval_acc.png"))
    except Exception as e:
        print(f"绘制总对比曲线失败：{e}")

    csv_path, md_path = _write_results_table(rows, exp_root)

    print("\n===== Done =====")
    print(f"输出目录: {exp_root}")
    print(f"对比表 CSV: {csv_path}")
    print(f"对比表 Markdown: {md_path}")
    print("对比曲线图: compare_loss.png / compare_eval_acc.png")


def _load_series_csv(csv_path: str, col: str):
    # 简易 CSV 读取：只读指定列
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


if __name__ == "__main__":
    main()

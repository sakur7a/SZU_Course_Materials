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
# -*- coding: utf-8 -*-
try:
    from pip._internal import main
    main(["install", "opencv-python", "matplotlib"])
except:
    raise ValueError("Please install opencv-python and matplotlib anually.")
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import shutil
import random
import numpy as np

from mindspore import context, Tensor
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import get_rank, init, get_group_size
from mindspore.train.serialization import export

def switch_precision(net, data_type, config):
    if config.platform == "Ascend":
        net.to_float(data_type)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)


def context_device_init(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)

    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                            save_graphs=False)
        if config.run_distribute:
            context.set_auto_parallel_context(device_num=config.rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, all_reduce_fusion_config=[140])
            init()
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


def set_context(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            save_graphs=False)
    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            device_id=config.device_id, save_graphs=False)
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=config.platform, save_graphs=False)


def delWithCmd(path):
    path = os.path.abspath(path)
    try:
        if os.path.isfile(path):
            cmd = 'del "'+ path + '" /F'
            os.system(cmd)
        if os.path.isdir(path):
            for f in os.listdir(path):
                delWithCmd(os.path.join(path, f))
    except Exception as e:
        print(e)

def export_mindir(net, name):
    input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    net.set_train(mode=False)
    path = os.path.abspath(f"{name}.mindir")
    export(net, Tensor(input_np), file_name=path, file_format='MINDIR')
    print(f"export {name} MINDIR file at {path}")

def prepare_ckpt(config):
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0')
    try:
        print("remove pre checkpoint")
        shutil.rmtree(save_ckpt_path)
    except Exception as e:
        delWithCmd(save_ckpt_path)
    if not os.path.isdir(save_ckpt_path):
        os.mkdir(save_ckpt_path)

def read_img(img_path, config):
    img_s = cv2.imread(img_path)
    h, w, c = img_s.shape
    new_w = int(500 / h * w)
    img_s = cv2.resize(img_s, (new_w, 500))
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_height = img_s.shape[0]
    img_width = img_s.shape[1]
    img = cv2.resize(img_s, (config.image_width, config.image_height))

    mean = np.array([0.406 * 255, 0.456 * 255, 0.485 * 255]).reshape((1, 1, 3))
    std = np.array([0.225 * 255, 0.224 * 255, 0.229 * 255]).reshape((1, 1, 3))
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)).reshape((1, 3, config.image_width, config.image_height)).astype(np.float32)
    return img_s, img, img_width, img_height

def _infer_label_index_from_path(img_path, config):
    """从图片路径中推断真实标签索引。

    约定：
        验证集通常按 ImageFolder 格式组织：.../eval/<class_name>/xxx.jpg。
        Kaggle 猫狗数据集常见类别名为 Cat/Dog（不区分大小写）。

    Returns:
        int|None: 若能推断则返回 0/1 等索引，否则返回 None。
    """
    try:
        cls_name = os.path.basename(os.path.dirname(img_path))
    except Exception:
        return None
    cls_name = (cls_name or "").strip().lower()
    # 尽量兼容各种命名
    if "cat" in cls_name:
        return 0
    if "dog" in cls_name:
        return 1
    return None

def _get_prob_and_pred(output_np):
    """从网络输出中获得预测类别与置信度。

    output_np: shape (1, num_classes)
    """
    x = output_np.reshape(-1)
    pred_idx = int(np.argmax(x))
    # 如果看起来像 softmax 概率（0~1 且和接近 1），直接使用
    if np.all(x >= 0) and np.all(x <= 1) and abs(float(np.sum(x)) - 1.0) < 1e-3:
        conf = float(np.max(x))
        return pred_idx, conf
    # 否则做一次 softmax 得到概率
    x_max = np.max(x)
    ex = np.exp(x - x_max)
    p = ex / np.sum(ex)
    conf = float(np.max(p))
    return pred_idx, conf

def predict_from_net(net, img_list, config, show_title="test", save_path=None, show=True):
    """对给定图片列表进行预测，并以网格方式可视化。

    Args:
        net: MindSpore 网络，输入为 (1, 3, H, W)，输出为类别概率/logits。
        img_list (list[str]): 图片路径列表。
        config: 配置对象，要求包含 config.map（类别索引到名称的映射）。
        show_title (str): 图像窗口总标题。
        save_path (str|None): 若不为 None，则把网格图保存为 PNG（可用于实验报告截图）。
        show (bool): 是否弹出窗口显示。
    """
    # 网格布局：为了让图片更清晰，默认优先采用“每行 5 张”的布局。
    # - 20 张（猫10+狗10） => 4x5
    # - 10 张            => 2x5
    # - 其它数量          => 尽量 5 列，行数向上取整
    n = len(img_list)
    if n >= 20:
        rows, cols = 4, 5
    elif n >= 10:
        rows, cols = 2, 5
    else:
        cols = 5
        rows = max((n + cols - 1) // cols, 1)

    # 画布尺寸：适当增大以提升每张子图的可读性
    plt.figure(figsize=(cols * 4.0, rows * 3.6))
    if show_title:
        plt.suptitle(show_title)

    # 统计准确率（若能从路径推断真值标签）
    total_with_label = 0
    correct_with_label = 0

    # 用于标识错误图片：红色边框
    try:
        import matplotlib.patches as patches
    except Exception:
        patches = None

    for i, file_path in enumerate(img_list):
        if i >= rows * cols:
            break
        img_s, img, img_width, img_height = read_img(file_path, config)
        output = net(Tensor(img)).asnumpy()
        pred_index, conf = _get_prob_and_pred(output)
        pred_cls = config.map[pred_index] if hasattr(config, "map") else str(pred_index)

        true_index = _infer_label_index_from_path(file_path, config)
        is_correct = None
        true_cls = None
        if true_index is not None and hasattr(config, "map") and true_index < len(config.map):
            true_cls = config.map[true_index]
            is_correct = (true_index == pred_index)
            total_with_label += 1
            if is_correct:
                correct_with_label += 1

        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(np.squeeze(img_s))
        # 标题：展示预测与置信度；若能推断真值，再展示真值并标注对错
        if true_cls is not None:
            title_text = f"P:{pred_cls} {conf:.2f}\nT:{true_cls}"
        else:
            title_text = f"P:{pred_cls} {conf:.2f}"

        if is_correct is False:
            plt.title(title_text, color='red')
            if patches is not None:
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(rect)
        elif is_correct is True:
            plt.title(title_text, color='green')
        else:
            plt.title(title_text)

        plt.xticks([])
        plt.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        save_path = os.path.abspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # 提高保存分辨率，保证网格图放大后仍清晰
        plt.savefig(save_path, dpi=300)
        print(f"预测网格图已保存: {save_path}")

    # 打印预测准确率（仅统计能从路径推断真值的样本）
    if total_with_label > 0:
        acc = correct_with_label / total_with_label
        print(f"{show_title} 预测准确率: {correct_with_label}/{total_with_label} = {acc:.2%}")

    if show:
        plt.show()
    plt.close()

    # predict_decode(output, img_s, config, show_title, save_path=None)

def get_samples_from_eval_dataset(dataset_path, sample_nums=6, per_class=None):
    """从验证集中随机抽样。

    Args:
        dataset_path (str): 数据集根目录，要求包含 eval/<class>/...
        sample_nums (int): 总共随机抽取多少张（per_class 为 None 时生效）
        per_class (int|None): 若设置为整数，则从每个类别文件夹各抽取 per_class 张。

    Returns:
        list[str]: 图片路径列表
    """
    eval_root = os.path.join(dataset_path, "eval")
    if per_class is not None:
        all_paths = []
        for sub_dir in os.listdir(eval_root):
            cls_dir = os.path.join(eval_root, sub_dir)
            if not os.path.isdir(cls_dir):
                continue
            files = [os.path.join(cls_dir, fn) for fn in os.listdir(cls_dir)]
            if not files:
                continue
            k = min(int(per_class), len(files))
            all_paths.extend(random.sample(files, k))
        random.shuffle(all_paths)
        return all_paths

    dirs = []
    for sub_dir in os.listdir(eval_root):
        cls_dir = os.path.join(eval_root, sub_dir)
        if not os.path.isdir(cls_dir):
            continue
        for file_name in os.listdir(cls_dir):
            dirs.append(os.path.join(cls_dir, file_name))
    k = min(int(sample_nums), len(dirs))
    return random.sample(dirs, k)

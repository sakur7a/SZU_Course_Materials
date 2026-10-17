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
import time
import os
import random
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.train.model import Model
from mindspore.train.serialization import save_checkpoint

from .mobilenetV2 import (
    MobileNetV2Backbone,
    MobileNetV2Head,
    MobileNetV2HeadDropout,
    MobileNetV2HeadMLP,
    mobilenet_v2,
)
from .utils import delWithCmd

class CrossEntropyWithLabelSmooth(_Loss):
    """
    CrossEntropyWith LabelSmooth.

    Args:
        smooth_factor (float): smooth factor. Default is 0.
        num_classes (int): number of classes. Default is 1000.

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss

class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))

def acc(feature, label, model):
    correct_num = 0
    total_num = 0
    for j in range(len(feature)):
        x = Tensor(feature[j])
        y = label[j]
        y_pred = model.predict(x).asnumpy()
        indices = y_pred.argmax(axis=1)
        result = (np.equal(indices, y) * 1).reshape(-1)
        correct_num += result.sum()
        total_num += result.shape[0]
    return correct_num / total_num

def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """load checkpoint into network."""
    param_dict = load_checkpoint(pretrain_ckpt_path)
    load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False

def define_net(config, activation="None"):
    backbone_net = MobileNetV2Backbone()
    head_variant = str(getattr(config, "head_variant", "baseline")).lower()
    if head_variant in ("baseline", "linear"):
        head_net = MobileNetV2Head(
            input_channel=backbone_net.out_channels,
            num_classes=config.num_classes,
            has_dropout=False,
        )
    elif head_variant in ("dropout", "dropout_linear"):
        keep_prob = float(getattr(config, "head_dropout_keep_prob", 0.8))
        head_net = MobileNetV2HeadDropout(
            input_channel=backbone_net.out_channels,
            num_classes=config.num_classes,
            keep_prob=keep_prob,
        )
    elif head_variant in ("mlp", "mlp2"):
        hidden_dim = int(getattr(config, "head_hidden_dim", 256))
        keep_prob = float(getattr(config, "head_dropout_keep_prob", 0.8))
        head_net = MobileNetV2HeadMLP(
            input_channel=backbone_net.out_channels,
            hidden_dim=hidden_dim,
            num_classes=config.num_classes,
            keep_prob=keep_prob,
        )
    else:
        raise ValueError(f"Unknown head_variant: {head_variant}")
    net = mobilenet_v2(backbone_net, head_net, activation)
    return backbone_net, head_net, net

def get_networks(net, loss, opt):
    network = WithLossCell(net, loss)
    network = TrainOneStepCell(network, opt)
    network.set_train()
    model = Model(net)
    return network, model

def train(network, model, net, data, config):
    """训练 head 网络，并在训练过程中记录指标。

    说明：
        当前项目采用“先抽取 backbone 特征、再训练 head”的方式。
        这里的 network 是 TrainOneStepCell(WithLossCell(net, loss), opt)，
        model 是 mindspore.train.Model(net) 用于 predict 计算准确率。

    Returns:
        dict: 训练历史，用于绘图/写报告。
            - epoch_loss: list[float]，每个 epoch 的平均 loss
            - train_acc: list[float]，每个 epoch 的训练集准确率
            - eval_acc: list[float]，每个 epoch 的验证集准确率
            - best: dict，最好 epoch 与综合 acc
    """
    history = {"epoch_loss": [], "train_acc": [], "eval_acc": [], "best": None}
    best_acc = {"epoch":0, "acc":0}
    early_stop_patience = int(getattr(config, "early_stop_patience", 5))
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0')
    if not os.path.isdir(save_ckpt_path):
        os.mkdir(save_ckpt_path)
    for epoch in range(config.epoch_size):
        train_features, train_labels, eval_features, eval_labels = data
        step_size = len(train_features)
        idx_list = list(range(step_size))
        random.shuffle(idx_list)
        epoch_start = time.time()
        losses = []
        for j in idx_list:
            feature = Tensor(train_features[j])
            label = Tensor(train_labels[j])
            losses.append(network(feature, label).asnumpy())
        # 当前 epoch 的平均 loss（用于打印 + 绘图）
        avg_loss = float(np.mean(np.array(losses))) if losses else float('nan')
        train_acc = acc(train_features, train_labels, model)
        eval_acc = acc(eval_features, eval_labels, model)

        history["epoch_loss"].append(avg_loss)
        history["train_acc"].append(float(train_acc))
        history["eval_acc"].append(float(eval_acc))

        epoch_mseconds = (time.time() - epoch_start) * 1000
        per_step_mseconds = epoch_mseconds / step_size
        print(
            "epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}, acc: {:5.3f}, train acc: {:5.3f}" \
            .format(epoch + 1, config.epoch_size, step_size, epoch_mseconds, per_step_mseconds, avg_loss,
                    eval_acc, train_acc))
        acc_f = (eval_acc * 2 + train_acc) / 3
        if acc_f >= best_acc["acc"]:
            print("update best acc:", acc_f)
            bast_acc_path = os.path.join(save_ckpt_path, "mobilenetv2_best.ckpt")
            delWithCmd(bast_acc_path)
            save_checkpoint(net, bast_acc_path)
            best_acc = {"epoch": epoch, "acc": acc_f}
            history["best"] = dict(best_acc)
        # early stop：当最佳指标在连续 N 个 epoch 内未更新则停止训练
        # - early_stop_patience = 0 表示关闭 early stop
        if early_stop_patience > 0 and (epoch - best_acc["epoch"]) > early_stop_patience:
            # best_acc['epoch'] 是 0-based，这里打印成 1-based 便于对齐日志/表格
            print(f"early stop! the best epoch is {best_acc['epoch'] + 1}")
            # load the parameters with best acc into net from pre_training checkpoint
            load_ckpt(net, os.path.join(save_ckpt_path, "mobilenetv2_best.ckpt"))
            break

    # 若训练过程中从未更新过 best，则至少给一个初始值
    if history["best"] is None:
        history["best"] = dict(best_acc)
    return history

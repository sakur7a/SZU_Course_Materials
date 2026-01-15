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
create train or eval dataset.
"""
import os
import numpy as np
import multiprocessing
from mindspore import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

def create_dataset(dataset_path, do_train, config, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    cores = max(min(multiprocessing.cpu_count(), 8), 1)
    ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=cores, shuffle=True)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((resize_height, resize_width))
    rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        batch_size = config.batch_size
    else:
        batch_size = 1
    trans = [decode_op, resize_op, normalize_op, change_swap_op]
    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(operations=trans, input_columns="image", num_parallel_workers=cores)
    ds = ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=cores)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds


def extract_features(net, dataset_path, config, cache_tag="default"):
    """使用 backbone 网络对数据集抽取特征并缓存到本地。

    说明：
        本项目训练时会先抽取 MobileNetV2 backbone 的特征并保存为 .npy，
        随后只训练 head（分类层），从而显著加速训练。

    为了支持“迁移学习 vs 随机初始化”的对比实验：
        不同 backbone 权重会产生不同特征，因此需要用 cache_tag 区分缓存目录，
        避免两次实验误用同一份 features 导致对比失真。

    Args:
        net: 用于抽取特征的网络（通常是 backbone）。
        dataset_path (str): 数据集根目录，要求包含 train/ 和 eval/。
        config: 配置对象。
        cache_tag (str): 特征缓存标签，用于区分不同实验，例如 "pretrain" / "scratch"。

    Returns:
        ((train_feature, train_label, eval_feature, eval_label), step_size)
    """
    print("start cache feature!")
    features_folder = os.path.join(dataset_path, f"features_{cache_tag}")

    train_dataset = create_dataset(dataset_path=os.path.join(dataset_path, "train"), do_train=True, config=config)
    eval_dataset = create_dataset(dataset_path=os.path.join(dataset_path, "eval"), do_train=False, config=config)

    train_size = train_dataset.get_dataset_size()
    eval_size = eval_dataset.get_dataset_size()
    if train_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")
    if os.path.exists(features_folder):
        train_features = np.load(os.path.join(features_folder, f"train_feature.npy"))
        train_labels = np.load(os.path.join(features_folder, f"train_label.npy"))
        eval_features = np.load(os.path.join(features_folder, f"eval_feature.npy"))
        eval_labels = np.load(os.path.join(features_folder, f"eval_label.npy"))
        return (train_features, train_labels, eval_features, eval_labels), train_size
    os.mkdir(features_folder)
    model = Model(net)
    train_feature = []
    train_labels = []
    train_imgs = train_size * config.batch_size
    for i, data in enumerate(train_dataset.create_dict_iterator()):
        image = data["image"]
        label = data["label"]
        feature = model.predict(Tensor(image))
        train_feature.append(feature.asnumpy())
        train_labels.append(label.asnumpy())
        percent = round(i / train_size * 100., 2)
        print(f'training feature cached [{i * config.batch_size}/{train_imgs}] {str(percent)}% ', end='\r', flush=True)
    np.save(os.path.join(features_folder, f"train_feature"), np.array(train_feature))
    np.save(os.path.join(features_folder, f"train_label"), np.array(train_labels))
    print(f'training feature cached [{train_imgs}/{train_imgs}] 100%  \ntrain feature cache finished!', flush=True)

    eval_feature = []
    eval_labels = []
    for i, data in enumerate(eval_dataset.create_dict_iterator()):
        image = data["image"]
        label = data["label"]
        feature = model.predict(Tensor(image))
        eval_feature.append(feature.asnumpy())
        eval_labels.append(label.asnumpy())
        percent = round(i / eval_size * 100., 2)
        print(f'evaluation feature cached [{i}/{eval_size}] {str(percent)}% ', end='\r')
    np.save(os.path.join(features_folder, f"eval_feature"), np.array(eval_feature))
    np.save(os.path.join(features_folder, f"eval_label"), np.array(eval_labels))
    print(f'evaluation feature cached [{eval_size}/{eval_size}] 100%  \neval feature cache finished!')
    print(f"saved feature features_folder")
    return (np.array(train_feature), np.array(train_labels), np.array(eval_feature), np.array(eval_labels)), train_size

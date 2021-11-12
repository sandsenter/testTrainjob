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
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

# download and put Mnist dataset into ./Data/train and ./Data/test directories
# python train.py --device_target=CPU > log_train.txt 2>&1

import os
import argparse
import moxing as mox
from config import mnist_cfg as cfg
from dataset import create_dataset
from lenet import LeNet5
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.common import set_seed

parser = argparse.ArgumentParser(description='MindSpore Lenet Example')

# define 2 parameters for running on modelArts
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='./data')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default='./model')

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'GPU', 'CPU'],
    help='device where the code will be implemented (default: Ascend)')

parser.add_argument('--dataset_path',
                    type=str,
                    default="./Data",
                    help='path where the dataset is saved')
parser.add_argument('--save_checkpoint_path',
                    type=str,
                    default="./ckpt",
                    help='if is test, must provide\
                    path where the trained ckpt file')
parser.add_argument('--epoch_size',
                    type=int,
                    default=5,
                    help='Training epochs.')

set_seed(1)

if __name__ == "__main__":

    args = parser.parse_args()

    # Copy obs_file to local
    obs_data_url = args.data_url
    args.data_url = '/home/work/user-job-dir/inputs/data/'
    obs_train_url = args.train_url
    args.train_url = '/home/work/user-job-dir/outputs/model/'
    try:
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      args.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            obs_data_url, args.data_url) + str(e))

    args.dataset_path = args.data_url
    args.save_checkpoint_path = args.train_url

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.dataset_path, "train"),
                              cfg.batch_size)
    if ds_train.get_dataset_size() == 0:
        raise ValueError(
            "Please check dataset size > 0 and batch_size <= dataset size")

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    if args.device_target != "Ascend":
        model = Model(network,
                      net_loss,
                      net_opt,
                      metrics={"accuracy": Accuracy()})
    else:
        model = Model(network,
                      net_loss,
                      net_opt,
                      metrics={"accuracy": Accuracy()},
                      amp_level="O2")

    config_ck = CheckpointConfig(
        save_checkpoint_steps=cfg.save_checkpoint_steps,
        keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory=args.save_checkpoint_path,
                                 config=config_ck)

    print("============== Starting Training ==============")
    epoch_size = cfg['epoch_size']
    if (args.epoch_size):
        epoch_size = args.epoch_size
        print('epoch_size is: ', epoch_size)

    model.train(epoch_size,
                ds_train,
                callbacks=[time_cb, ckpoint_cb,
                           LossMonitor()])

    # Upload model to obs
    try:
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(args.train_url,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.train_url,
                                                       obs_train_url) + str(e))

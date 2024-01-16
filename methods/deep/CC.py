# Copyright (c) 2023 Yuxuan Shao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This method reproduction refers to the following repository:
# https://github.com/Yunfan-Li/Contrastive-Clustering/tree/main
from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset
from utils import config
from .base import DeepMethod
from .backbone.CC_resnet import get_resnet, Network
from .loss.CC_loss import InstanceLoss, ClusterLoss
from .utils.CC_utils import Transforms


class CC(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.batch_size = self.cfg.get("CC", "batch_size")
        self.image_size = self.cfg.get("CC", "image_size")
        self.epochs = self.cfg.get("CC", "epochs")
        self.learn_rate = self.cfg.get("CC", "learn_rate")
        self.weight_decay = self.cfg.get("CC", "learn_rate")
        self.instance_temperature = self.cfg.get("CC", "instance_temperature")
        self.cluster_temperature = self.cfg.get("CC", "cluster_temperature")
        resnet = get_resnet(self.cfg.get("CC", "resnet"))
        self.model = Network(resnet, self.cfg.get("CC", "feature_dim"), self.n_clusters).to(self.device)

    def forward(self, x):
        pass

    def pretrain(self):
        self.dataset.pretrain()
        return None

    def train_model(self):
        self.dataset.clustering()

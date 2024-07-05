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
import torch.nn as nn
from logging import Logger

from utils import config
from metrics import Metrics
from datasetLoader import ClusteringDataset


class DeepMethod(nn.Module):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__()
        self.dataset = dataset
        self.description = description
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.get("global", "device")
        self.metrics = Metrics()
        self.workers = cfg.get("global", "workers")
        self.weight_dir = cfg.get("global", "weight_dir")
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            self.n_clusters = dataset.num_classes
        else:
            self.n_clusters = cfg.get("global", "n_clusters")
            assert type(
                self.n_clusters) is int, "n_clusters should be of type int"
            assert self.n_clusters > 0, "n_clusters should be larger than 0"

    def forward(self, x):
        """
        This is the forward pass of the model

        If you don't use it, you can just leave it aside.
        """
        return x

    def pretrain(self):
        """
        This method is used to pretrain the model.

        It should return the hidden layer representation of the whole dataset.

        In order to draw the pretraining loss figure, the `total_loss` should be passed to the `update_pretrain_loss` method of the metrics.
        Any other loss you want to draw should also be passed to the `update_pretrain_loss` method of the metrics with the format `update_pretrain_loss(..., loss_name=loss_value)`.

        For those methods that do not need pretraining, just return None.
        """
        raise NotImplementedError

    def train_model(self):
        """
        This method is used to train the model.

        It should return the predicted labels, features.

        In order to draw the clustering loss figure, the `total_loss` should be passed to the `update_loss` method of the metrics.
        Any other loss you want to draw should also be passed to the `update_loss` method of the metrics with the format `update_loss(..., loss_name=loss_value)`.
       
        If ground truth is available, the `y_true` should be passed to the `update` method of the metrics.
        """
        raise NotImplementedError

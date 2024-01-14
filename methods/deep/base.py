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

class DeepMethod(nn.Module):
    def __init__(self, dataset, description, logger: Logger, cfg: config):
        super().__init__()
        self.dataset = dataset
        self.description = description
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.get("global", "device")
        self.metrics = Metrics()
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            self.n_clusters = dataset.num_classes
        else:
            self.n_clusters = cfg.get("global", "n_clusters")
            assert type(
                self.n_clusters) is int, "n_clusters should be of type int"
            assert self.n_clusters > 0, "n_clusters should be larger than 0"
    
    def forward(self, x):
        raise NotImplementedError
    # This is the forward pass of the model
    
    def pretrain(self):
        raise NotImplementedError
    # This method is used to pretrain the model.
    # It should return the hidden layer representation of the whole dataset.
    # There must be any loss update in the metrics by `update_pretrain_loss` to draw the pretrain loss figure.
    # For those methods that do not need pretraining, just return None.
    
    def train_model(self):
        raise NotImplementedError
    # This method is used to train the model.
    # It should return the predicted labels, features.
    # There must be a `total_loss` meaning the total model loss update in the metrics by `update_loss` to draw the clustering loss figure.
    # If ground truth is available, the `y_true` should be passed to the `update` method of the metrics.

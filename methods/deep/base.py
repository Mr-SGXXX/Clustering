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
    
    def forward(self, x):
        raise NotImplementedError
    # This is the forward pass of the model
    
    def pretrain(self):
        raise NotImplementedError
    # This method is used to pretrain the model.
    # It should return the hidden layer representation of the whole dataset.
    # There must be any loss update in the metrics by `update_pretrain_loss` to draw the pretrain loss figure.
    
    def train_model(self):
        raise NotImplementedError
    # This method is used to train the model.
    # It should return the predicted labels, features.
    # There must be a `total_loss` meaning the total model loss update in the metrics by `update_loss` to draw the clustering loss figure.
    # If ground truth is available, the `y_true` should be passed to the `update` method of the metrics.

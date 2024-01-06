import torch.nn as nn
from logging import Logger
from utils import config

class DeepMethod(nn.Module):
    def __init__(self, dataset, description, logger: Logger, cfg: config):
        super().__init__()
        self.dataset = dataset
        self.description = description
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.get("global", "device")
    
    def forward(self, x):
        raise NotImplementedError
    # this is the forward pass of the model
    
    def pretrain(self):
        raise NotImplementedError
    # this method is used to pretrain the model.
    # it should return the hidden layer representation of the whole dataset and the pretrain loss list.
    
    def train_model(self):
        raise NotImplementedError
    # this method is used to train the model.
    # it should return the predicted labels, features and metrics.
    # There must be a `total_loss` update in the metrics.

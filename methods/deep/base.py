import torch.nn as nn
from logging import Logger
from utils import config

class DeepMethod(nn.Module):
    def __init__(self, dataset, logger: Logger, cfg: config):
        super().__init__()
        self.dataset = dataset
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.get("global", "device")
    
    def forward(self, x):
        raise NotImplementedError
    
    def pretrain(self):
        raise NotImplementedError
    
    def train_model(self):
        raise NotImplementedError
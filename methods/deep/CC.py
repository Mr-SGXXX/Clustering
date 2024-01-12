from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from utils import config

from .base import DeepMethod

class CC(DeepMethod):
    def __init__(self, dataset, description, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        pass

    def forward(self, x):
        pass

    def pretrain(self):
        pass

    def train_model(self):
        pass
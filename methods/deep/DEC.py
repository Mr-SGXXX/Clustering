from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from utils import config

from .base import DeepMethod
from .backbone.DEC_AE import DEC_AE
from .backbone.layers import ClusteringLayer
from .utils.DEC_utils import target_distribution


class DEC(DeepMethod):
    def __init__(self, dataset, logger: Logger, cfg: config):
        super().__init__(dataset, logger, cfg)
        self.input_dim = dataset.input_dim
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            self.n_clusters = dataset.num_classes
        else:
            self.n_clusters = cfg.get("global", "n_clusters")
            assert type(
                self.n_clusters) is int, "n_clusters should be of type int"
            assert self.n_clusters > 0, "n_clusters should be larger than 0"
        self.encoder_dims = cfg.get("DEC", "encoder_dims")
        self.decoder_dims = cfg.get("DEC", "decoder_dims")
        self.hidden_dim = cfg.get("DEC", "hidden_dim")
        self.alpha = cfg.get("DEC", "alpha")
        self.batch_size = cfg.get("DEC", "batch_size")
        self.pretrain_lr = cfg.get("DEC", "pretrain_learn_rate")
        self.lr = cfg.get("DEC", "learn_rate")

        self.ae = DEC_AE(self.input_dim, self.encoder_dims, self.decoder_dims, self.hidden_dim).to(self.device)
        self.clustering_layer = ClusteringLayer(self.n_clusters, self.hidden_dim, self.alpha).to(self.device)
    
    def forward(self, x):
        x_bar, z = self.ae(x)
        q = self.clustering_layer(z)
        return x_bar, q, z
    
    def pretrain(self):
        optimizer = optim.Adam(self.ae.parameters(), lr=self.pretrain_lr)
        optimizer = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
        train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        for i in range(len(self.encoder_dims) + 1):
            for j in range(len(self.ae.encoder)):
                if j > i * 3:
                    break
                self.ae.encoder[j].requires_grad_(False)
                self.ae.decoder[-(j+1)].requires_grad_(False)
            for it in range(50000):
                for data, _ in train_loader:
                    data = data.to(self.device)
                    x_bar, _ = self.ae(data)
                    loss = nn.MSELoss()(x_bar, data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
        return 

    def train_model(self):
        pass
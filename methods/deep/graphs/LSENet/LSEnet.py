# MIT License

# Copyright (c) 2023-2024 Yuxuan Shao

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
from logging import Logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import from_networkx, negative_sampling
from torch_scatter import scatter_sum
from geoopt.optim import RiemannianAdam
from torch import optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config

from .LSEnet_model import HyperSE
from .LSEnet_utils import index2adjacency

from methods.deep.base import DeepMethod
# This method reproduction refers to the following repository:
# https://github.com/ZhenhHuang/LSEnet


class LSEnet(DeepMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.hidden_dim = cfg.get("LSEnet", "hidden_dim")
        self.hidden_dim_enc = cfg.get("LSEnet", "hidden_dim_enc")
        self.height = cfg.get("LSEnet", "height")
        self.temperature = cfg.get("LSEnet", "temperature")
        self.embed_dim = cfg.get("LSEnet", "embed_dim")
        self.dropout = cfg.get("LSEnet", "dropout")
        self.nonlin = cfg.get("LSEnet", "nonlin")
        self.decay_rate = cfg.get("LSEnet", "decay_rate")
        self.max_nums = cfg.get("LSEnet", "max_nums")
        self.lr_pre = cfg.get("LSEnet", "lr_pre")
        self.w_decay = cfg.get("LSEnet", "w_decay")
        self.pre_epochs = cfg.get("LSEnet", "pre_epochs")
        
        self.dataset.use_full_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        self.data = {
            "feature": graph.x,
            "edge_index": edge_index,
            "degrees": scatter_sum(torch.ones(edge_index.shape[1]), edge_index[0]),
            "weight": torch.ones(edge_index.shape[1]),
            "neg_edge_index": negative_sampling(edge_index),
            "adj": index2adjacency(graph.num_nodes, edge_index, torch.ones(edge_index.shape[1]), is_sparse=True)
        }
        self.model:HyperSE = HyperSE(in_features=self.input_dim,
                            hidden_features=self.hidden_dim,
                            hidden_dim_enc=self.hidden_dim_enc,
                            num_nodes=graph.num_nodes,
                            height=self.height, temperature=self.temperature,
                            embed_dim=self.embed_dim, dropout=self.dropout,
                            nonlin=self.nonlin,
                            decay_rate=self.decay_rate,
                            max_nums=self.max_nums).to(self.device)
        
    def pretrain(self):
        optimizer_pre = RiemannianAdam(self.model.parameters(), lr=self.lr_pre,
                                           weight_decay=self.w_decay)
        with tqdm(range(1, self.pre_epochs), desc="Pretraining") as pbar:
            for epoch in pbar:
                self.model.train()
                loss = self.model.loss(self.data, self.data['edge_index'], self.data['neg_edge_index'], self.device, pretrain=True)
                optimizer_pre.zero_grad()
                loss.backward()
                optimizer_pre.step()
                self.logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
                self.metrics.update_pretrain_loss(total_loss=loss.item())
        
        return None
    
    def clustering(self):
        pass
        
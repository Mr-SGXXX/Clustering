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
import torch.optim.lr_scheduler as lr_scheduler
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.dmon_pool import DMoNPooling
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch import optim
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from community import best_partition as louvain
from tqdm import tqdm
import numpy as np
import os
import networkx as nx
from torch_geometric.utils import to_networkx

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .DMoN_gcn import GCN_skip

# This method reproduction refers to the following repository:
# https://github.com/google-research/google-research/tree/master/graph_embedding/dmon

class DMoN(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.hidden_dims = cfg.get("DMoN", "hidden_dims")
        self.collapse_weight = cfg.get("DMoN", "collapse_weight")
        self.dropout_rate = cfg.get("DMoN", "dropout_rate")
        self.num_epochs = cfg.get("DMoN", "num_epochs")
        self.learn_rate = cfg.get("DMoN", "learn_rate")
        
        if isinstance(self.hidden_dims, int):
            self.hidden_dims = [self.hidden_dims]
        self.gcns = nn.ModuleList()
        input_dim = self.input_dim
        for n_channels in self.hidden_dims:
            self.gcns.append(GCN_skip(input_dim, n_channels))
            input_dim = n_channels
        self.dmon_pooling = DMoNPooling(input_dim, self.n_clusters, self.dropout_rate)
        
        self.to(self.device)
            
    
    def clustering(self):
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos")
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        es_count = 0
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_index = SparseTensor.to_torch_sparse_coo_tensor(edge_index)
        
        y_pred_last = np.zeros(graph.y.shape[0])
        with tqdm(range(self.num_epochs), desc="Training") as epoch_loader:
            for epoch in epoch_loader:
                optimizer.zero_grad()
                z, pred, spectral_loss, collapse_loss = self(x, edge_index)
                collapse_loss = self.collapse_weight * collapse_loss
                
                total_loss = spectral_loss + collapse_loss
                total_loss.backward()
                optimizer.step()
                
                
                self.eval()
                with torch.no_grad():
                    z, pred, spectral_loss, collapse_loss = self(x, edge_index)
                self.train()
                y_pred = pred.data.cpu().numpy().argmax(1)
                
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z, y_true=graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
                    
                self.metrics.update_loss(total_loss=total_loss.item(), spectral_loss=spectral_loss.item(), collapse_loss=collapse_loss.item())
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                y_pred_last = y_pred
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}")
                # if delta_label < 1e-4:
                #     es_count += 1
                # else:
                #     es_count = 0
                # if es_count >= 3:
                #     self.logger.info(f"Early stopping at epoch {epoch} with delta_label= {delta_label:.4f}")
                #     break
                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "F1_macro": f1_macro,
                    "F1_weighted": f1_weighted,
                    "Delta_label": delta_label,
                    "Delta_NMI": delta_nmi,
                })
        self.eval()
        with torch.no_grad():
            z, pred, spectral_loss, collapse_loss = self(x, edge_index)
        y_pred = pred.data.cpu().numpy().argmax(1)
        self.logger.info(f"Clustering Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        self.metrics.update_loss(total_loss=total_loss.item(), spectral_loss=spectral_loss.item(), collapse_loss=collapse_loss.item())
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        return y_pred, z
        
    
    def forward(self, x, edge_index:SparseTensor):
        for gcn in self.gcns:
            x = gcn(x, edge_index)
            x = F.selu(x)
        pred, _, _, spectral_loss, _, cluster_loss = self.dmon_pooling(x, edge_index.to_dense())
        pred = torch.squeeze(pred)
        return x, pred, spectral_loss, cluster_loss
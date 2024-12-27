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
from torch_sparse import SparseTensor
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_undirected, add_remaining_self_loops, dropout_adj
from torch import optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config

from .GRACE_model import Model, Encoder, drop_feature

from methods.deep.base import DeepMethod
# This method reproduction refers to the following repository:
# https://github.com/CRIPAC-DIG/GRACE


class GRACE(DeepMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.num_hidden = cfg.get("GRACE_2020", "num_hidden")
        self.num_proj_hidden = cfg.get("GRACE_2020", "num_proj_hidden")
        self.activation = cfg.get("GRACE_2020", "activation")
        if self.activation == "relu":
            self.activation = nn.ReLU()
        elif self.activation == "prelu":
            self.activation = nn.PReLU()
        else:
            raise ValueError("Activation function not supported")
        self.base_model = cfg.get("GRACE_2020", "base_model")
        if self.base_model == "GCNConv":
            self.base_model = GCNConv
        self.num_layers = cfg.get("GRACE_2020", "num_layers")
        
        self.drop_edge_rate_1 = cfg.get("GRACE_2020", "drop_edge_rate_1")
        self.drop_edge_rate_2 = cfg.get("GRACE_2020", "drop_edge_rate_2")
        self.drop_feature_rate_1 = cfg.get("GRACE_2020", "drop_feature_rate_1")
        self.drop_feature_rate_2 = cfg.get("GRACE_2020", "drop_feature_rate_2")
        self.tau = cfg.get("GRACE_2020", "tau")
        self.lr = cfg.get("GRACE_2020", "learn_rate")
        self.num_epochs = cfg.get("GRACE_2020", "num_epochs")
        self.weight_decay = cfg.get("GRACE_2020", "weight_decay")
    
    def clustering(self):
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        graph = T.NormalizeFeatures()(graph)
        encoder = Encoder(self.input_dim, self.num_hidden, self.activation, self.base_model, self.num_layers).to(self.device)
        model:Model = Model(encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        edge_index:SparseTensor = graph.edge_index
        edge_index = edge_index.to_torch_sparse_coo_tensor().coalesce()
        edge_index = edge_index.indices().to(self.device)
        x = graph.x.to(self.device)
        
        y_pred_last = np.zeros_like(graph.y.cpu().numpy())
        with tqdm(range(self.num_epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                model.train()
                optimizer.zero_grad()
                edge_index_1 = dropout_adj(edge_index, p=self.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(edge_index, p=self.drop_edge_rate_2)[0]
                x_1 = drop_feature(x, self.drop_feature_rate_1)
                x_2 = drop_feature(x, self.drop_feature_rate_2)
                z1 = model(x_1, edge_index_1)
                z2 = model(x_2, edge_index_2)

                total_loss = model.loss(z1, z2, batch_size=0)
                total_loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    embedding = model(x, edge_index)
                self.metrics.update_loss(total_loss=total_loss.item())
                y_pred = KMeans(n_clusters=self.n_clusters).fit_predict(embedding.cpu().detach().numpy())
                
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                y_pred_last = y_pred
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, embedding, y_true=graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
                
                self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}")
            
                epoch_loader.set_postfix({
                    "acc": acc,
                    "nmi": nmi,
                    "ari": ari,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "delta_label": delta_label,
                    "delta_nmi": delta_nmi
                })
        return y_pred, embedding
        
  

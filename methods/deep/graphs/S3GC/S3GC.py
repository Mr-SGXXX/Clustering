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
import scipy.sparse as sp
from torch_geometric.utils import to_networkx
from gensim.models import Word2Vec

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .S3GC_utils import norm_adj, compute_diffusion_matrix
from .S3GC_model import S3GC_Model
# This method reproduction refers to the following repository:
# https://drive.google.com/drive/folders/18B_eWbdVhOURZhqwoBSsyryb4WsiYLQK

class S3GC(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.hidden_dims = cfg.get("S3GC", "hidden_dims")
        self.walk_length = cfg.get("S3GC", "walk_length")
        self.walks_per_node = cfg.get("S3GC", "walks_per_node")
        self.batch_size = cfg.get("S3GC", "batch_size")
        self.lr = cfg.get("S3GC", "learn_rate")
        self.epochs = cfg.get("S3GC", "epochs")
        self.big_model = cfg.get("S3GC", "big_model")
    
    def clustering(self):
        self.dataset.use_full_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        graph = graph.to(self.device)
        
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
        
        y = graph.y
        normalized_A = norm_adj(torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=self.device), (graph.x.size(0), graph.x.size(0))))
        AX = torch.sparse.mm(normalized_A, graph.x)
        SX = compute_diffusion_matrix(normalized_A, graph.x, niter=3)
        del graph, normalized_A
        
        model:S3GC_Model = S3GC_Model(AX.size(-1), self.hidden_dims, AX.size(0), edge_index,
                           self.hidden_dims, self.walk_length, self.walk_length,
                           self.walks_per_node, big_model=self.big_model, p=1.0, q=1.0).to(self.device)
    
        loader = model.loader(batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        mapping:torch.Tensor = torch.zeros(AX.size(0), dtype=torch.int64).to(self.device)
        
        model.train()   
        y_pred_last = np.zeros_like(y.cpu().numpy())
        with tqdm(range(self.epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                total_loss = 0
                with tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False) as batch_loader:
                    for i, (pos_rw, neg_rw) in enumerate(batch_loader):
                        optimizer.zero_grad()
                        pos_rw = pos_rw.to(self.device)
                        neg_rw = neg_rw.to(self.device)
                        unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
                        mapping.scatter_(0, unique, torch.arange(unique.size(0)).to(self.device))
                        model.update_B(F.embedding(unique, AX), F.embedding(unique, SX), unique)
                        loss = model.loss(pos_rw, neg_rw, mapping)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        batch_loader.set_postfix({
                            "loss": loss.item()
                        })
                        
                total_loss /= len(loader)
                embedding = model.get_embedding(AX, SX)
                self.metrics.update_loss(total_loss=total_loss)
                y_pred = KMeans(n_clusters=self.n_clusters).fit(embedding.cpu().detach().numpy()).predict(embedding.cpu().detach().numpy())
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                y_pred_last = y_pred
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, embedding, y_true=y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=y)
                
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
                
        
        
        
    

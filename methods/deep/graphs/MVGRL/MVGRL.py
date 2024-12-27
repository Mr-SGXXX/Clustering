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

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .MVGRL_model import Model, LogReg
from .MVGRL_utils import preprocess_features, normalize_adj, sparse_mx_to_torch_sparse_tensor, compute_ppr, compute_heat
# This method reproduction refers to the following repository:
# https://github.com/kavehhassani/mvgrl
# MVGRL is a representation learning method that aims to learn a good representation of the graph data.
# In this project, the learned representation will be input to kmeans clustering algorithm to cluster the data.

class MVGRL(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.nb_epochs = cfg.get("MVGRL", "nb_epochs")
        self.patience = cfg.get("MVGRL", "patience")
        self.lr = cfg.get("MVGRL", "learn_rate")
        self.l2_coef = cfg.get("MVGRL", "l2_coef")
        self.hid_units = cfg.get("MVGRL", "hid_units")
        self.sample_size = cfg.get("MVGRL", "sample_size")
        self.batch_size = cfg.get("MVGRL", "batch_size")
        self.diffusion_type = cfg.get("MVGRL", "diffusion_type")
        
    def clustering(self):
        self.weight_dir = os.path.join(self.weight_dir, "MVGRL")
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        adj = graph.edge_index.to_torch_sparse_coo_tensor()
        indices = adj.coalesce().indices().cpu().numpy()
        values = adj.coalesce().values().cpu().numpy()
        shape = adj.size()
        adj = sp.csr_matrix((values, indices), shape=shape)
        features = preprocess_features(graph.x).to(self.device)
        
        if self.diffusion_type == "heat":
            diff = compute_heat(adj.toarray())
        elif self.diffusion_type == "ppr":
            diff = compute_ppr(adj.toarray())
        else:
            raise NotImplementedError("Unknown diffusion type")
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).toarray()
        
        adj_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj)).to(self.device)
        diff_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff)).to(self.device)
        
        model:Model = Model(self.input_dim, self.hid_units).to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        
        lbl_1 = torch.ones(self.batch_size, self.sample_size * 2).to(self.device)
        lbl_2 = torch.zeros(self.batch_size, self.sample_size * 2).to(self.device)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        
        b_xent = nn.BCEWithLogitsLoss()
        
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        with tqdm(range(self.nb_epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                
                idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1, self.batch_size)           
                ba = torch.cat([sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj[i: i + self.sample_size, i: i + self.sample_size])).unsqueeze(0) for i in idx]).to(self.device)
                bd = torch.cat([sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff[i: i + self.sample_size, i: i + self.sample_size])).unsqueeze(0) for i in idx]).to(self.device)
                bf = torch.cat([features[i: i + self.sample_size].unsqueeze(0) for i in idx])
            
                idx = np.random.permutation(self.sample_size)
                shuf_fts = bf[:, idx, :]
                 
                model.train()
                optimiser.zero_grad()

                logits, _, _ = model(bf, shuf_fts, ba, bd, True, None, None, None) 

                loss = b_xent(logits, lbl)
                
                loss.backward()
                optimiser.step()
                
                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    if not os.path.exists(self.weight_dir):
                        os.makedirs(self.weight_dir)
                    torch.save(model.state_dict(), os.path.join(self.weight_dir, "best_dgi.pkl"))
                else:
                    cnt_wait += 1
                
                if cnt_wait == self.patience:
                    self.logger.info(f'Early stopping at epoch: {epoch}, best epoch: {best_t}')
                    break
                
                model.eval()
                h = model.embed(features, adj_tensor, diff_tensor, True, None)[0].squeeze(0).cpu().detach().numpy()
                y_pred = KMeans(n_clusters=self.n_clusters, random_state=self.cfg["global"]["seed"]).fit_predict(h)
                model.train()
                
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, h, y_true=graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
                
                self.metrics.update_loss(total_loss=loss.item())
                
                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "F1_macro": f1_macro,
                    "F1_weighted": f1_weighted
                })
                
                
        model.load_state_dict(torch.load(os.path.join(self.weight_dir, "best_dgi.pkl")))
        model.eval()
        h = model.embed(features, adj_tensor, diff_tensor, True, None)[0].squeeze(0).cpu().detach().numpy()
        y_pred = KMeans(n_clusters=self.n_clusters, random_state=self.cfg["global"]["seed"]).fit_predict(h)
        
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, h, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        self.metrics.update_loss(total_loss=best.item())
        self.logger.info(f"Final Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
    
        return y_pred, h
        
        
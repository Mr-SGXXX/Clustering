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
import scipy as sp
from torch_geometric.utils import to_networkx

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .DGCluster_model import GNN
from .DGCluster_loss import loss_fn
# This method reproduction refers to the following repository:
# https://github.com/pyrobits/DGCluster/blob/master/main.py

class DGCluster(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.base_model = cfg.get("DGCluster", "base_model")
        self.lam = cfg.get("DGCluster", "lam")
        self.alp = cfg.get("DGCluster", "alp")
        self.epochs = cfg.get("DGCluster", "epochs")
        self.ground_truth_for_train = cfg.get("DGCluster", "ground_truth_for_train")
        self.pre_clustering_method = cfg.get("DGCluster", "pre_clustering_method")
        
    
    def clustering(self): 
        self.dataset.use_full_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        model = GNN(self.dataset.input_dim, 64, base_model=self.base_model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.epochs)
        model.train()
        graph = graph.to(self.device)
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        num_nodes = graph.x.shape[0]
        num_edges = (edge_index.shape[1])
        sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(self.device)
        
        if self.ground_truth_for_train:
            know_labels = torch.from_numpy(self.dataset.label)
            oh_labels = F.one_hot(know_labels, num_classes=self.n_clusters)
            if self.dataset.unlabel_length > 0:
                unknown_labels = torch.zeros((self.dataset.unlabel_length, self.n_clusters), dtype=torch.long)
                oh_labels = torch.cat([oh_labels, unknown_labels], dim=0)
        else:
            # In the paper, the authors wrote:
            #   When those ground-truth information is not available, we
            #   can also leverage traditional structure-based graph partitioning heuristics,
            #   such as Louvain (Blondel et al. 2008), and treat their generated clusters as the node labels.
            #   We can also use traditional clustering algorithms, such as KMeans, Spectral Clustering, and Birch,
            if self.pre_clustering_method == "Louvain":
                Graph = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph).to_undirected()
                partition = louvain(Graph, resolution=0.5)
                pre_clustering_label = torch.tensor([partition[i] if i in partition else -1 for i in range(num_nodes)], dtype=torch.long)
            elif self.pre_clustering_method == "KMeans":
                pre_clustering_label = KMeans(n_clusters=self.n_clusters, random_state=self.cfg["global"]["seed"]).fit_predict(self.dataset.data)
            elif self.pre_clustering_method == "SpectralClustering":
                pre_clustering_label = SpectralClustering(n_clusters=self.n_clusters, affinity="nearest_neighbors").fit_predict(self.dataset.data)
            elif self.pre_clustering_method == "Birch":
                pre_clustering_label = Birch(n_clusters=self.n_clusters, threshold=0.5).fit_predict(self.dataset.data)
            num_classes = len(np.unique(pre_clustering_label))
            pre_clustering_label[pre_clustering_label == -1] = num_classes - 1
            oh_labels = F.one_hot(torch.tensor(pre_clustering_label, dtype=torch.long), num_classes=num_classes)
            if -1 in pre_clustering_label:
                oh_labels = oh_labels[:-1]
        oh_labels = oh_labels.to(self.device)
        
        with tqdm(range(self.epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                optimizer.zero_grad()
                output = model(graph)
                
                m_loss, aux_loss, reg_loss = loss_fn(output, edge_index, oh_labels, sparse_adj, degree, self.lam, self.alp)
                total_loss = m_loss + aux_loss + reg_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                
                self.metrics.update_pretrain_loss(total_loss=total_loss.item(), modularity_loss=m_loss.item(), aux_loss=aux_loss.item(), reg_loss=reg_loss.item())
                
        model.eval()
        x = model(graph)
        y_pred = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)

        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, x, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        
        self.logger.info(f"Final Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        
        
        return y_pred, x
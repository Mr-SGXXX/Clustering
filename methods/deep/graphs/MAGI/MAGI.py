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
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch import optim
from time import time
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import numpy as np
import os


from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .MAGI_utils import get_mask, get_sim, scale
from .MAGI_Sampler import NeighborSampler
from .MAGI_model import Encoder, Model
from .MAGI_kmeans_gpu import kmeans
# This method reproduction refers to the following repository:
# https://github.com/EdisonLeeeee/MAGI

class MAGI(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.base_model = cfg.get("MAGI", "base_model")
        self.hidden_dims = cfg.get("MAGI", "hidden_dims")
        self.hidden_dims = [self.hidden_dims] if not isinstance(self.hidden_dims, list) else self.hidden_dims
        self.projection_dims = cfg.get("MAGI", "projection_dims")
        self.sizes_neighbor = cfg.get("MAGI", "sizes_neighbor", default=[10, 10])
        self.wt = cfg.get("MAGI", "wt") # number of random walks
        self.wl = cfg.get("MAGI", "wl") # depth of random walks
        self.tau = cfg.get("MAGI", "tau") # temperature
        self.dropout = cfg.get("MAGI", "dropout")
        self.lr = cfg.get("MAGI", "learn_rate")
        self.weight_decay = cfg.get("MAGI", "weight_decay")
        self.epochs = cfg.get("MAGI", "epochs")
        self.batch_size = cfg.get("MAGI", "batch_size")
        self.kmeans_batch = cfg.get("MAGI", "kmeans_batch", default=-1)
        self.negative_slope = cfg.get("MAGI", "negative_slope") # negative slope of leaky relu
        self.clustering_method = cfg.get("MAGI", "clustering_method") # clustering method for representation
        self.max_duration = cfg.get("MAGI", "max_duration")
        self.kmeans_device = cfg.get("MAGI", "kmeans_device", default="cpu")
        
        
    def clustering(self):
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        x = graph.x.to(self.device)
        N, E = graph.num_nodes, graph.edge_index.nnz()
        self.logger.info(f"Graph data: {N} nodes, {E} edges and {self.dataset.input_dim} features.")
        
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
        
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N))
        adj.fill_value_(1.)
        
        encoder = Encoder(self.dataset.input_dim, hidden_channels=self.hidden_dims, base_model=self.base_model, dropout=self.dropout, ns=self.negative_slope)
        encoder = encoder.to(self.device)
        
        model = Model(encoder, self.hidden_dims[-1], self.projection_dims, tau=self.tau).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.base_model == "GraphSAGE": # GraphSAGE for big graph
            train_loader = NeighborSampler(edge_index, adj,
                                            is_train=True,
                                            node_idx=None,
                                            wt=self.wt,
                                            wl=self.wl,
                                            sizes=self.sizes_neighbor,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=self.workers)
            
            test_loader = NeighborSampler(edge_index, adj,
                                            is_train=False,
                                            node_idx=None,
                                            sizes=self.sizes_neighbor,
                                            batch_size=10000,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=self.workers)
            ts_train = time()
            stop_pos = False
            with tqdm(range(1, self.epochs+1), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
                for epoch in epoch_loader:
                    model.train()
                    total_loss = total_examples = 0

                    for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
                        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                        if len(self.sizes_neighbor) == 1:
                            adjs = [adjs]
                        adjs = [adj.to(self.device) for adj in adjs]

                        adj_ = get_mask(adj_batch)
                        optimizer.zero_grad()
                        out = model(x[n_id].to(self.device), adjs=adjs)
                        out = F.normalize(out, p=2, dim=1)
                        loss = model.loss(out, adj_)

                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        total_examples += batch_size
                        if (time.time() - ts_train) // 60 >= self.max_duration:
                            self.logger.info(f"Maximum training time is exceeded at epoch {epoch}.")
                            stop_pos = True
                            break
                    if epoch % 5 == 1:
                        self.logger.info(f"Epoch {epoch}/{self.epochs} Loss: {total_loss/total_examples}")
                    if stop_pos:
                        break
            with torch.no_grad():
                model.eval()
                z = []
                for (batch_size, n_id, adjs), _, batch in tqdm(test_loader):
                    if len(self.sizes_neighbor) == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    out = model(x[n_id].to(self.device), adjs=adjs)
                    z.append(out.detach().cpu().float())
                z = torch.cat(z, dim=0)
                z = F.normalize(z, p=2, dim=1)
                
        elif self.base_model == "GCN":
            # in the official implementation, the edge_index used in gcn is:
            # edge_index = data.edge_index.to(device)
            # where data is the orginal graph data
            edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices().to(self.device)
            batch = torch.LongTensor(list(range(N)))
            batch, adj_batch = get_sim(batch, adj, wt=self.wt, wl=self.wl)
            mask = get_mask(adj_batch)
        
            # train
            with tqdm(range(1, self.epochs + 1), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
                for epoch in epoch_loader:
                    model.train()
                    optimizer.zero_grad()
                    out = model(x, edge_index)
                    out = scale(out)
                    out = F.normalize(out, p=2, dim=1)
                    loss = model.loss(out, mask)
                    loss.backward()
                    optimizer.step()
                    self.metrics.update_pretrain_loss(loss=loss.item())
                    if epoch % 10 == 1:
                        self.logger.info(f"Epoch {epoch}/{self.epochs} Loss: {loss.item()}")

            # eval
            with torch.no_grad():
                model.eval()
                z = model(x, edge_index)
                z = scale(z)
                z = F.normalize(z, p=2, dim=1)
        else:
            raise NotImplementedError(f"Clustering with base model {self.base_model} is not implemented.")
    
        if self.clustering_method == "KMeans":
            if self.kmeans_device == "cpu":
                Cluster = KMeans(n_clusters=self.n_clusters, max_iter=10000, n_init=20)
                y_pred = Cluster.fit_predict(z.data.cpu().numpy())
            elif self.kmeans_device == "gpu":
                y_pred, _ = kmeans(z, self.n_clusters, batch_size=self.kmeans_batch, device=self.device, tol=1e-4)
                y_pred = y_pred.cpu().numpy()
            else:
                raise NotImplementedError(f"Kmeans device {self.kmeans_device} is not implemented.")
        elif self.clustering_method == "SpectralClustering":
            Cluster = SpectralClustering(
            n_clusters=self.n_clusters, affinity='precomputed', random_state=0)
            feature = z.data.cpu().numpy()
            f_adj = np.matmul(feature, np.transpose(feature))
            y_pred = Cluster.fit_predict(f_adj)
        else:
            raise NotImplementedError(f"Clustering method {self.clustering_method} is not implemented.")
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        
        self.logger.info(f"Final Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        return y_pred, z
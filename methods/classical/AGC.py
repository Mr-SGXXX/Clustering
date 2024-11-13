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

from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import typing
from logging import Logger
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor, matmul

from utils import config
from datasetLoader import ClusteringDataset

from .base import ClassicalMethod

# This method reproduction refers to the following repository:
# https://github.com/karenlatong/AGC-master

class AGC(ClassicalMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.max_iter = cfg.get("AGC", "max_iterations")
        
    def clustering(self):
        # load data
        graph_data:Data = self.dataset.to_graph()
        feature = graph_data.x.to(self.device)
        if isinstance(graph_data.edge_index, torch.Tensor):
            edge_index = graph_data.edge_index.to(self.device)
            edge_attr = graph_data.edge_attr.to(self.device)
            edge_index = to_dense_adj(edge_index, edge_attr=edge_attr).squeeze(0)
        elif isinstance(graph_data.edge_index, SparseTensor):
            edge_index = graph_data.edge_index.to_device(self.device)
        adj_normalized:SparseTensor = preprocess_adj(edge_index)
        e = SparseTensor.eye(graph_data.num_nodes, graph_data.num_nodes).to_device(adj_normalized.device())
        adj_normalized = adj_normalized + e
        adj_normalized.set_value(adj_normalized.storage.value().to(self.device) / 2)
        
        tt = 0
        intra_list = []
        intra_list.append(10000)
        while True:
            tt += 1
            feature = matmul(adj_normalized, feature)
            f = feature.detach().cpu().numpy()
            u, s, v = svds(f, k=self.n_clusters, which='LM')
            
            predict_label = KMeans(n_clusters=self.n_clusters, random_state=self.cfg["global"]["seed"]).fit(u).predict(u)
            intraD = square_dist(predict_label, f)
            intra_list.append(intraD)
            
            if intra_list[tt] > intra_list[tt - 1] or tt > self.max_iter:
                self.logger.info(f"Best power: {tt-1}")
                return predict_label, u
            
        
def normalize_adj(adj:typing.Union[torch.Tensor, SparseTensor], type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        if isinstance(adj, SparseTensor):
            adj = adj.to_torch_sparse_coo_tensor()
        elif isinstance(adj, torch.Tensor):
            adj = adj.to_sparse_coo()
        # rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        rowsum = torch.sparse.sum(adj, 1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        indices = torch.arange(0, d_inv_sqrt.size(0)).to(adj.device)
        indices = torch.stack([indices, indices]).to(adj.device)  # [2, N]
        d_mat_inv_sqrt = torch.sparse_coo_tensor(indices, d_inv_sqrt, (d_inv_sqrt.size(0), d_inv_sqrt.size(0)))
        adj_normalized = d_mat_inv_sqrt.matmul(adj).matmul(d_mat_inv_sqrt)
        return SparseTensor.from_torch_sparse_coo_tensor(adj_normalized)
    elif type == 'rw':
        adj:torch.Tensor = adj.to_sparse_coo()
        # rowsum = np.array(adj.sum(1))
        # d_inv = np.power(rowsum, -1.0).flatten()
        # d_inv[np.isinf(d_inv)] = 0.
        # d_mat_inv = sp.diags(d_inv)
        # adj_normalized = d_mat_inv.dot(adj)
        rowsum = torch.sparse.sum(adj, 1).to_dense()
        d_inv = torch.pow(rowsum, -1.0)
        d_inv[torch.isinf(d_inv)] = 0.
        indices = torch.arange(0, d_inv.size(0)).to(adj.device)
        indices = torch.stack([indices, indices]).to(adj.device)
        d_mat_inv = torch.sparse_coo_tensor(indices, d_inv, (d_inv.size(0), d_inv.size(0)))
        adj_normalized = d_mat_inv.matmul(adj)
        return SparseTensor.from_torch_sparse_coo_tensor(adj_normalized)

def preprocess_adj(adj, type='sym', loop=True) -> SparseTensor:
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        # adj = adj + sp.eye(adj.shape[0])
        if isinstance(adj, SparseTensor):
            adj = adj + SparseTensor.eye(adj.size(0), adj.size(0)).to_device(adj.device())
        elif isinstance(adj, torch.Tensor):
            adj += torch.eye(adj.size(0)).to(adj.device)
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized

def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist
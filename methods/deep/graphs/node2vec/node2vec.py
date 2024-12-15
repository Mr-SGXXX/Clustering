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

from .node2vec_Graph import Graph
# This method reproduction refers to the following repository:
# https://github.com/aditya-grover/node2vec
# node2vec is a representation learning method that aims to learn a good representation of the graph data.
# In this project, the learned representation will be input to kmeans clustering algorithm to cluster the data.

class node2vec(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.dimensions = cfg.get("node2vec", "dimensions")
        self.walk_length = cfg.get("node2vec", "walk_length")
        self.num_walks = cfg.get("node2vec", "num_walks")
        self.window_size = cfg.get("node2vec", "window_size")
        self.iter = cfg.get("node2vec", "iter")
        self.p = cfg.get("node2vec", "p")
        self.q = cfg.get("node2vec", "q")
        self.directed = cfg.get("node2vec", "directed")
    
    def clustering(self):
        self.dataset.use_full_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        graph = graph.to(self.device)
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        num_nodes = graph.x.shape[0]
        num_edges = (edge_index.shape[1])
        sparse_adj = sp.csr_matrix((np.ones(num_edges), edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
        nx_G = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph)
        if not self.directed:
            nx_G = nx_G.to_undirected()
            
        G = Graph(nx_G, self.directed, self.p, self.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        
        model = Word2Vec(walks, vector_size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=8, epochs=self.iter)
        
        embeddings = torch.Tensor([model.wv[str(node)] for node in range(num_nodes)])
        
        kmeans = KMeans(n_clusters=self.dataset.num_classes, n_init=200)
        
        y_pred = kmeans.fit_predict(embeddings)
        
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, embeddings, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        
        self.logger.info(f"Clustering Results: ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f},  F1_macro: {f1_macro:.4f}, F1_weighted: {f1_weighted:.4f}")
        
        return y_pred, embeddings
                
        
        
        
    

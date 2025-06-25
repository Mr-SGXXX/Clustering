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

from .GraphMAE_model import PreModel
# This method reproduction refers to the following repository:
# https://github.com/THUDM/GraphMAE
# GraphMAE is a representation learning method that aims to learn a good representation of the graph data.
# In this project, the learned representation will be input to kmeans clustering algorithm to cluster the data.

class GraphMAE(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.max_epoch = cfg.get("GraphMAE", "max_epoch")
        self.num_hidden = cfg.get("GraphMAE", "num_hidden")
        self.num_layers = cfg.get("GraphMAE", "num_layers")
        self.encoder_type = cfg.get("GraphMAE", "encoder_type")
        self.decoder_type = cfg.get("GraphMAE", "decoder_type")
        self.num_nheads = cfg.get("GraphMAE", "num_nheads")
        self.num_out_heads = cfg.get("GraphMAE", "num_out_heads")
        self.activation = cfg.get("GraphMAE", "activation")
        self.in_drop = cfg.get("GraphMAE", "in_drop")
        self.attn_drop = cfg.get("GraphMAE", "attn_drop")
        self.negative_slope = cfg.get("GraphMAE", "negative_slope") 
        self.residual = cfg.get("GraphMAE", "residual")
        self.norm = cfg.get("GraphMAE", "norm")
        self.mask_rate = cfg.get("GraphMAE", "mask_rate")
        self.loss_fn = cfg.get("GraphMAE", "loss_fn")
        self.drop_edge_rate = cfg.get("GraphMAE", "drop_edge_rate")
        self.replace_rate = cfg.get("GraphMAE", "replace_rate")
        self.lr = cfg.get("GraphMAE", "learn_rate")
        self.weight_decay = cfg.get("GraphMAE", "weight_decay")
        self.patience = cfg.get("GraphMAE", "patience")
        self.use_scheduler = cfg.get("GraphMAE", "use_scheduler")
        self.alpha_l = cfg.get("GraphMAE", "alpha_l")
        self.concat_hidden = cfg.get("GraphMAE", "concat_hidden")
        
        
        
    def clustering(self):
        self.weight_dir = os.path.join(self.weight_dir, "GraphMAE")
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        features = graph.x.to(self.device)
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        edge_index = to_undirected(add_remaining_self_loops(edge_index)[0]).to(self.device)
        
        model:PreModel = PreModel(
            in_dim=self.input_dim,
            num_hidden=self.num_hidden, 
            num_layers=self.num_layers,
            nhead=self.num_nheads,
            nhead_out=self.num_out_heads,
            activation=self.activation,
            feat_drop=self.in_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type,
            mask_rate=self.mask_rate,
            norm=self.norm,
            loss_fn=self.loss_fn,
            drop_edge_rate=self.drop_edge_rate,
            replace_rate=self.replace_rate,
            alpha_l=self.alpha_l,
            concat_hidden=self.concat_hidden,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.use_scheduler:
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / self.max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        with tqdm(range(self.max_epoch), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                model.train()
                optimizer.zero_grad()

                loss, _ = model(features, edge_index)
                
                loss.backward()
                optimizer.step()
                
                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    if not os.path.exists(self.weight_dir):
                        os.makedirs(self.weight_dir)
                    torch.save(model.state_dict(), os.path.join(self.weight_dir, "best_GraphMAE.pkl"))
                else:
                    cnt_wait += 1
                
                if cnt_wait == self.patience:
                    self.logger.info(f'Early stopping at epoch: {epoch}, best epoch: {best_t}')
                    break
                
                model.eval()
                h = model.embed(features, edge_index).cpu().detach().numpy()
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
                
                
        
        model.load_state_dict(torch.load(os.path.join(self.weight_dir, "best_GraphMAE.pkl")))
        model.eval()
        h = model.embed(features, edge_index).cpu().detach().numpy()
        y_pred = KMeans(n_clusters=self.n_clusters, random_state=self.cfg["global"]["seed"]).fit_predict(h)
        
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, h, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        self.metrics.update_loss(total_loss=best.item())
        self.logger.info(f"Final Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
    
        return y_pred, h
        
        
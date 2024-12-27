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
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch import optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config

from .GRACE_model import GRACE_Model
from .GRACE_utils import target_distribution

from methods.deep.base import DeepMethod
# This method reproduction refers to the following repository:
# https://github.com/yangji9181/GRACE?utm_source=catalyzex.com


class GRACE(DeepMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.encoder_dims = cfg.get("GRACE_2017", "encoder_dims")
        self.decoder_dims = cfg.get("GRACE_2017", "decoder_dims")
        self.embedding_dim = cfg.get("GRACE_2017", "embedding_dim")
        self.transition_function = cfg.get("GRACE_2017", "transition_function")
        self.random_walk_step = cfg.get("GRACE_2017", "random_walk_step")
        self.dropout_rate = cfg.get("GRACE_2017", "dropout_rate")
        self.lambda_r = cfg.get("GRACE_2017", "lambda_r")
        self.lambda_c = cfg.get("GRACE_2017", "lambda_c")
        self.alpha = cfg.get("GRACE_2017", "alpha")
        self.lambda_ = cfg.get("GRACE_2017", "lambda_")
        self.lr = cfg.get("GRACE_2017", "learn_rate")
        self.pre_epoch = cfg.get("GRACE_2017", "pre_epoch")
        self.epoch = cfg.get("GRACE_2017", "epoch")
        self.step = cfg.get("GRACE_2017", "step")
        self.bn_flag = cfg.get("GRACE_2017", "bn_flag")
        self.epsilon = cfg.get("GRACE_2017", "epsilon")
        
        
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        self.x = graph.x.to(self.device)
        T, RI, RW = self.init_graph(graph, alpha=self.alpha, lambda_=self.lambda_)
        self.T = T.to(self.device)
        self.RI = RI.to(self.device)
        self.RW = RW.to(self.device)

        self.model = GRACE_Model(self.input_dim, self.encoder_dims, self.decoder_dims, self.embedding_dim,
                                 self.n_clusters, self.T, self.RI, self.RW, self.transition_function,
                                 self.random_walk_step, self.dropout_rate, self.bn_flag, self.epsilon).to(self.device)
    
    def pretrain(self):
        self.dataset.use_full_data()
        self.model.train()
         
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        with tqdm(range(self.pre_epoch), desc="Pretrain AE", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                x_rec, _, z = self.model(self.x)
                loss = F.binary_cross_entropy_with_logits(x_rec, self.x, reduction='mean')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss = loss.item()
                if (epoch+1) % 10 == 0:
                    self.logger.info("Pretrain Epoch {} loss={:.4f}".format(
                        epoch + 1, total_loss))
                epoch_loader.set_postfix({
                    "loss": total_loss
                })
                self.metrics.update_pretrain_loss(total_loss=(total_loss))
        self.logger.info("Pretraining finished!")
        return z
    
    def clustering(self):
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        es_count = 0
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.x)[2]
        self.model.train()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())

        acc, nmi, ari, f1_macro, f1_weighted, _, _ = evaluate(y_pred, self.dataset.label)
        self.logger.info(f"Pretrain Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        y_pred_last = y_pred
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        with tqdm(range(self.epoch * self.step), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                if epoch % self.step == 0:
                # update_interval
                    self.model.eval()
                    with torch.no_grad():
                        _, tmp_q, _ = self.model(self.x)
                    self.model.train()
                    p = target_distribution(tmp_q.data)
                    
                    
                x_bar, q, z = self.model(self.x)

                loss_r = F.binary_cross_entropy_with_logits(x_bar, self.x, reduction='mean') * self.lambda_r
                # loss_c = torch.mean(p * torch.log(p / q)) * self.lambda_c
                loss_c = F.kl_div(q.log(), p, reduction='batchmean') * self.lambda_c

                total_loss = loss_r + loss_c
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                self.model.eval()
                with torch.no_grad():
                    x_bar, q, z = self.model(self.x)
                self.model.train()
                y_pred = q.data.cpu().numpy().argmax(1)   #Z
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z, y_true=graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
                
                self.metrics.update_loss(total_loss=total_loss.item(), loss_r=loss_r.item(), loss_c=loss_c.item())
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}")
                # if delta_label < self.tol:
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
        self.model.eval()
        with torch.no_grad():
            _, p, z = self.model(self.x)
        y_pred = torch.argmax(p, dim=1).data.cpu().numpy()
        self.logger.info(f"Clustering Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        self.metrics.update_loss(total_loss=total_loss.item(), loss_r=loss_r.item(), loss_c=loss_c.item())
        if self.cfg.get("global", "record_sc"):
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z, y_true=graph.y)
        else:
            _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=graph.y)
        return y_pred, z
    
    def init_graph(self, graph:GraphData, alpha = 0.9, lambda_ = 0.1):
        edge_index = graph.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        # edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])        
        # Construct RI, RW, and T values
        edge_index = edge_index.cpu().numpy()  # Convert edge_index to numpy for easier processing
        num_nodes = edge_index.max() + 1  # Determine the number of nodes based on edge_index
        edges = {v: set() for v in range(num_nodes)}  # Initialize adjacency list

        # Populate adjacency list from edge_index
        for src, dst in edge_index.T:
            edges[src].add(dst)
            edges[dst].add(src)

        # Ensure self-loops are present
        for v in range(num_nodes):
            edges[v].add(v)

        # Initialize lists to store indices and values for T, RI, and RW
        indices = []
        T_values, RI_values, RW_values = [], [], []

        # Construct indices and values for T, RI, and RW
        for v, ns in edges.items():
            indices.append(np.array([v, v]))
            T_values.append(1.0 / len(ns))
            RI_values.append(1.0 - alpha / len(ns))
            RW_values.append(1.0 - (1.0 - lambda_) / len(ns))
            for n in ns:
                if v != n:
                    indices.append(np.array([v, n]))
                    T_values.append(1.0 / len(ns))
                    RI_values.append(-alpha / len(ns))
                    RW_values.append(-(1.0 - lambda_) / len(ns))


        # Convert lists to numpy arrays
        indices = np.array(indices)
        T_values = np.asarray(T_values, dtype=np.float32)
        # RI_values = np.asarray(RI_values, dtype=np.float32)
        # RW_values = np.asarray(RW_values, dtype=np.float32)
        
        RI:torch.Tensor = torch.linalg.inv(torch.sparse_coo_tensor(
            indices=(indices[:, 1], indices[:, 0]),
            values=torch.tensor(RI_values, dtype=torch.float32),
            size=(num_nodes, num_nodes)
        ).to_dense())

        RW:torch.Tensor = lambda_ * torch.linalg.inv(torch.sparse_coo_tensor(
            indices=(indices[:, 0], indices[:, 1]),
            values=torch.tensor(RW_values, dtype=torch.float32),
            size=(num_nodes, num_nodes)
        ).to_dense())
        RW = RW / torch.sum(RW, dim=0)
        T = torch.sparse_coo_tensor(indices=(indices[:, 0], indices[:, 1]), values=T_values, size=(num_nodes, num_nodes))
        return T, RI, RW

        
        
  

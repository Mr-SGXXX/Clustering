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
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data as GraphData
from torch import optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config

from .SDCN_layer import AE, GNNLayer
from .SDCN_utils import target_distribution, normalize

from methods.deep.base import DeepMethod
# This method reproduction refers to the following repository:
# https://github.com/bdy9527/SDCN


class SDCN(DeepMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.encoder_dims = cfg.get("SDCN", "encoder_dims")
        self.hidden_dim = cfg.get("SDCN", "hidden_dim")
        self.alpha = cfg.get("SDCN", "alpha")
        self.sigma = cfg.get("SDCN", "sigma")
        self.batch_size = cfg.get("SDCN", "pretrain_batch_size")
        self.pretrain_lr = cfg.get("SDCN", "pretrain_learn_rate")
        self.lr = cfg.get("SDCN", "learn_rate")
        self.train_max_epoch = cfg.get("SDCN", "train_max_epoch")
        self.tol = cfg.get("SDCN", "tol")

        self.ae = AE(500, 500, 2000, 2000, 500, 500, self.input_dim, self.hidden_dim).to(self.device)
        
        self.gnn_1 = GNNLayer(self.input_dim, 500).to(self.device)
        self.gnn_2 = GNNLayer(500, 500).to(self.device)
        self.gnn_3 = GNNLayer(500, 2000).to(self.device)
        self.gnn_4 = GNNLayer(2000, self.hidden_dim).to(self.device)
        self.gnn_5 = GNNLayer(self.hidden_dim, self.n_clusters).to(self.device)
        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, self.hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h1 = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        pred = F.softmax(h1, dim=1)
        
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1.0)
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return x_bar, q, pred, h
    
    def encode_dataset(self):
        self.eval()
        train_loader = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=self.workers)
        latent_list = []
        with torch.no_grad():
            for data, _, _ in tqdm(train_loader, desc="Encoding dataset", dynamic_ncols=True, leave=False):
                data = data.to(self.device)
                # _, z, _ = self.ae(data)
                z = self.ae(data)[4]
                latent_list.append(z)
        latent = torch.cat(latent_list, dim=0)
        self.train()
        return latent
    
    def pretrain(self):
        self.dataset.use_full_data()
        self.train()
        model = self.ae
        # print(model)
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        optimizer = optim.Adam(model.parameters(), lr=self.pretrain_lr)
        with tqdm(range(30), desc="Pretrain AE", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                total_loss = 0.
                for batch_idx, (x, _, _) in enumerate(train_loader):
                    x = x.to(self.device)
                    
                    x_bar = model(x)[0]
                    loss = F.mse_loss(x_bar, x)
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch+1) % 10 == 0:
                    self.logger.info("Pretrain Epoch {} loss={:.4f}".format(
                        epoch + 1, total_loss / (batch_idx + 1)))
                epoch_loader.set_postfix({
                    "loss": total_loss / (batch_idx + 1)
                })
                self.metrics.update_pretrain_loss(total_loss=(total_loss / (batch_idx + 1)))
        self.logger.info("Pretraining finished!")
        return self.encode_dataset()
    
    def clustering(self):
        self.dataset.use_label_data()
        graph:GraphData = self.dataset.to_graph(distance_type="NormCos", k=3)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # es_count = 0
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_index = normalize(edge_index)
        # print(edge_index)
        with torch.no_grad():
            z = self.ae(x)[4]
            
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())

        acc, nmi, ari, f1_macro, f1_micro, _, _ = evaluate(y_pred, self.dataset.label)
        self.logger.info(f"Pretrain Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_micro:.4f}")
        y_pred_last = y_pred
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        with tqdm(range(self.train_max_epoch), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                if epoch % 1 == 0:
                # update_interval
                    _, tmp_q, pred, h = self(x, edge_index)
                    p = target_distribution(tmp_q.data)
                
                    y_pred = pred.data.cpu().numpy().argmax(1)   #Z
                    
                x_bar, q, pred, _ = self(x, edge_index)
                    
                kl_loss = 0.1 * F.kl_div(q.log(), p, reduction='batchmean')
                ce_loss = 0.01 * F.kl_div(pred.log(), p, reduction='batchmean')
                re_loss = F.mse_loss(x_bar, x)

                total_loss = kl_loss + ce_loss + re_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                y_pred_last = y_pred
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_micro, _, _) = self.metrics.update(y_pred, h, y_true=graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_micro, _, _) = self.metrics.update(y_pred, y_true=graph.y)
                
                self.metrics.update_loss(total_loss=total_loss.item(), kl_loss=kl_loss.item(), ce_loss=ce_loss.item(), re_loss=re_loss.item())
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_micro:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}")
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
                    "F1_micro": f1_micro,
                    "Delta_label": delta_label,
                    "Delta_NMI": delta_nmi,
                })
        return y_pred, h
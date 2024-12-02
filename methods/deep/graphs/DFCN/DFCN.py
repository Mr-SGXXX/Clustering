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
from torch_geometric.nn.dense.dmon_pool import DMoNPooling
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
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA
from torch_geometric.utils import spmm
import torch_sparse

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .DFCN_IGAE import IGAE
from .DFCN_AE import AE
from .DFCN_utils import target_distribution, normalize
# This method reproduction refers to the following repository:
# https://github.com/WxTu/DFCN

class DFCN(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = cfg.get("DFCN", "input_dim")
        self.lambda_value = cfg.get("DFCN", "lambda_value")
        self.gamma_value = cfg.get("DFCN", "gamma_value")
        self.pretrain_learn_rate = cfg.get("DFCN", "pretrain_learn_rate")
        self.learn_rate = cfg.get("DFCN", "learn_rate")
        freedom_degree = cfg.get("DFCN", "freedom_degree")
        self.epochs = cfg.get("DFCN", "epochs")
        n_node = len(dataset)
        
        # preprocess the data
        data = dataset.data.numpy()
        # data = (data - data.mean(axis=0)) / data.std(axis=0)

        self.X_pca = PCA(n_components=self.input_dim).fit_transform(data)
        self.X_pca = torch.FloatTensor(self.X_pca)
        self.dataset.data = self.X_pca
        
        self.ae = AE(
            ae_n_enc_1=128,
            ae_n_enc_2=256,
            ae_n_enc_3=512,
            ae_n_dec_1=512,
            ae_n_dec_2=256,
            ae_n_dec_3=128,
            n_input=self.input_dim,
            n_z=20).to(self.device)

        self.gae = IGAE(
            gae_n_enc_1=128,
            gae_n_enc_2=256,
            gae_n_enc_3=20,
            gae_n_dec_1=20,
            gae_n_dec_2=256,
            gae_n_dec_3=128,
            n_input=self.input_dim).to(self.device)

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, 20), 0.5), requires_grad=True).to(self.device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, 20), requires_grad=True).to(self.device)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = freedom_degree
        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)
        
        self.graph = self.dataset.to_graph(distance_type="Heat", k=5, data_desc="PCA").to(self.device)
        self.graph.edge_index = normalize(self.graph.edge_index).to(self.device)
    
    
    def pretrain(self):
        """
        This pretrain method is following the description from the original paper:
        
        The training of the proposed DFCN includes three steps. 
        First, we pre-train the AE and IGAE independently for 30 iterations by minimizing the reconstruction loss functions. 
        Then, both subnetworks are integrated into a united framework for another 100 iterations. 
        Finally, with the learned centers of different clusters and under the guidance of the triplet self-supervised
        strategy, we train the whole network for at least 200 iterations until convergence.
        """
        
        edge_index:SparseTensor = self.graph.edge_index
        
        self.ae.train()
        self.gae.train()
        optimizer_ae = optim.Adam(self.ae.parameters(), lr=self.pretrain_learn_rate)
        optimizer_gae = optim.Adam(self.gae.parameters(), lr=self.pretrain_learn_rate)
        with tqdm(range(30), desc="Pretrain Stage 1") as pbar:
            for i in pbar:
                x_hat, _ = self.ae(self.graph.x)
                loss_rec_ae = F.mse_loss(x_hat, self.graph.x)
                _, z_hat, adj_hat = self.gae(self.graph.x, edge_index)
                loss_rec_gae = F.mse_loss(z_hat, spmm(edge_index, self.graph.x))
                loss_rec_gae_adj = F.mse_loss(adj_hat, edge_index.to_dense())
                
                loss_gae = loss_rec_gae + self.gamma_value * loss_rec_gae_adj
                
                optimizer_ae.zero_grad()
                optimizer_gae.zero_grad()
                loss_rec_ae.backward()
                loss_gae.backward()
                optimizer_ae.step()
                optimizer_gae.step()
                
                pbar.set_postfix({"loss_rec_ae": loss_rec_ae.item(), "loss_rec_gae": loss_rec_gae.item(), "loss_rec_gae_adj": loss_rec_gae_adj.item()})

                if i % 10 == 0:
                    self.logger.info(f"Pretrain Stage 1: Epoch {i}, Loss: {loss_rec_ae.item() + loss_rec_gae.item() + loss_rec_gae_adj.item()}")
                self.metrics.update_pretrain_loss(loss_rec_ae=loss_rec_ae.item(), loss_rec_gae=loss_rec_gae.item(), loss_rec_gae_adj=loss_rec_gae_adj.item())

        optimizer = optim.Adam(self.parameters(), lr=self.pretrain_learn_rate)
        with tqdm(range(100), desc="Pretrain Stage 2") as pbar:
            for i in pbar:
                x_hat, z_hat, adj_hat, _, _, _, _, _, _ = self.forward(self.graph)
                loss_rec_ae = F.mse_loss(x_hat, self.graph.x)
                loss_rec_gae = F.mse_loss(z_hat, spmm(edge_index, self.graph.x))
                loss_rec_gae_adj = F.mse_loss(adj_hat, edge_index.to_dense())
                
                loss = loss_rec_ae + loss_rec_gae + self.gamma_value * loss_rec_gae_adj
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({"loss": loss.item(), "loss_rec_ae": loss_rec_ae.item(), "loss_rec_gae": loss_rec_gae.item(), "loss_rec_gae_adj": loss_rec_gae_adj.item()})
                if i % 10 == 0:
                    self.logger.info(f"Pretrain Stage 2: Epoch {i}, Loss: {loss.item()}")
                self.metrics.update_pretrain_loss(loss_rec_ae=loss_rec_ae.item(), loss_rec_gae=loss_rec_gae.item(), loss_rec_gae_adj=loss_rec_gae_adj.item())
        
        return self.forward(self.graph)[-1]
            
            
    def clustering(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        edge_index:SparseTensor = self.graph.edge_index
        with torch.no_grad():
            x_hat, z_hat, adj_hat, _, _, _, _, _, z_tilde = self.forward(self.graph)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z_tilde.cpu().detach().numpy())
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        acc, nmi, ari, f1_macro, f1_weighted, _, _ = evaluate(y_pred, self.dataset.label)
        self.logger.info(f"Pretrain Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        y_pred_last = y_pred
        
        with tqdm(range(self.epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                x_hat, z_hat, adj_hat, _, _, q, q1, q2, z_tilde = self.forward(self.graph)

                tmp_q = q.data
                p = target_distribution(tmp_q)
                
                loss_ae = F.mse_loss(x_hat, self.graph.x)
                loss_w = F.mse_loss(z_hat, spmm(edge_index, self.graph.x))
                loss_a = F.mse_loss(adj_hat, edge_index.to_dense())
                loss_igae = loss_w + self.gamma_value * loss_a
                loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
                total_loss = loss_ae + loss_igae + self.lambda_value * loss_kl
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                y_pred = KMeans(n_clusters=self.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy()).labels_
                
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                y_pred_last = y_pred
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, z_tilde, y_true=self.graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=self.graph.y)
                
                self.metrics.update_loss(total_loss=total_loss.item(), loss_ae=loss_ae.item(), loss_igae=loss_igae.item(), loss_kl=loss_kl.item())
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}")
                    
                
                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "F1_macro": f1_macro,
                    "F1_weighted": f1_weighted,
                    "Delta_label": delta_label,
                    "Delta_NMI": delta_nmi,
                })

        return y_pred, z_tilde
        
        
    def forward(self, data):
        x, adj = data.x, data.edge_index
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde

        
        
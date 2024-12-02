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

from datasetLoader import ClusteringDataset
from utils.metrics import normalized_mutual_info_score as cal_nmi
from utils.metrics import evaluate
from utils import config
from methods.deep.base import DeepMethod

from .DCRN_layers import AE, IGAE, Readout
from .DCRN_utils import normalize_adj, diffusion_adj, remove_edge, gaussian_noised_feature, target_distribution
from .DCRN_loss import dicr_loss, reconstruction_loss, distribution_loss

# This method reproduction refers to the following repository:
# https://github.com/yueliu1999/DCRN/tree/main

class DCRN(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = cfg.get("DCRN", "input_dim")
        self.alpha_value = cfg.get("DCRN", "alpha_value")
        self.lambda_value = cfg.get("DCRN", "lambda_value")
        self.gamma_value = cfg.get("DCRN", "gamma_value")
        self.pretrain_learn_rate = cfg.get("DCRN", "pretrain_learn_rate")
        self.learn_rate = cfg.get("DCRN", "learn_rate")
        self.epochs = cfg.get("DCRN", "epochs")
        
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
        self.b = nn.Parameter(nn.init.constant_(torch.zeros(n_node, 20), 0.5), requires_grad=True).to(self.device)
        self.alpha = nn.Parameter(torch.zeros(1)).to(self.device)
        
        self.cluster_centers = nn.Parameter(torch.Tensor(self.n_clusters, 20), requires_grad=True).to(self.device)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

        self.R = Readout(K=self.n_clusters)
        
        
        self.graph = self.dataset.to_graph(distance_type="Heat", k=5, data_desc="PCA").to(self.device)
        self.A_norm = normalize_adj(self.graph.edge_index, self_loop=True, symmetry=False).to(self.device)
        self.Ad = diffusion_adj(self.graph.edge_index, mode="ppr", transport_rate=self.alpha_value).to(self.device)
        
    def pretrain(self):
        """
        This pretrain method is following the description from the original paper:
        
        Following DFCN (Tu et al. 2020), we frst pre-train the subnetworks independently with 
        at least 30 epochs by minimizing the reconstruction loss LREC. 
        Then both sub-networks are directly integrated into a united framework to obtain the
        initial clustering centers for another 100 epochs.
        """
        
        self.ae.train()
        self.gae.train()
        optimizer_ae = optim.Adam(self.ae.parameters(), lr=self.pretrain_learn_rate)
        optimizer_gae = optim.Adam(self.gae.parameters(), lr=self.pretrain_learn_rate)
        with tqdm(range(30), desc="Pretrain Stage 1") as pbar:
            for i in pbar:
                x_hat, _ = self.ae(self.graph.x)
                loss_rec_ae = F.mse_loss(x_hat, self.graph.x)
                _, z_hat, adj_hat = self.gae(self.graph.x, self.A_norm)
                loss_rec_gae = F.mse_loss(z_hat, spmm(self.A_norm, self.graph.x))
                loss_rec_gae_adj = F.mse_loss(adj_hat, self.A_norm.to_dense())
                
                loss_gae = loss_rec_gae + 0.1 * loss_rec_gae_adj
                
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
                # x_hat, z_hat, adj_hat, _, _, _, _, _, _ = self.forward(self.graph)
                # x_hat, z_hat, A_hat, _, _, _, _, Z, _, _ = self.forward(self.graph.x, self.A_norm, self.graph.x, self.A_norm)
                
                x, adj = self.graph.x, self.A_norm
                z_ae = self.ae.encoder(x)
                z_igae, z_igae_adj, _, _ = self.gae.encoder(x, adj)
                z_i = self.a * z_ae + self.b * z_igae
                z_l = spmm(adj, z_i)
                s = torch.mm(z_l, z_l.t())
                s = F.softmax(s, dim=1)
                z_g = torch.mm(s, z_l)
                z_tilde = self.alpha * z_g + z_l
                x_hat = self.ae.decoder(z_tilde)
                z_hat, z_hat_adj, _, _ = self.gae.decoder(z_tilde, adj)
                adj_hat = z_igae_adj + z_hat_adj
                
                loss_rec_ae = F.mse_loss(x_hat, self.graph.x)
                loss_rec_gae = F.mse_loss(z_hat, spmm(self.A_norm, self.graph.x))
                loss_rec_gae_adj = F.mse_loss(adj_hat, self.A_norm.to_dense())
                
                loss = loss_rec_ae + loss_rec_gae + 0.1 * loss_rec_gae_adj
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({"loss": loss.item(), "loss_rec_ae": loss_rec_ae.item(), "loss_rec_gae": loss_rec_gae.item(), "loss_rec_gae_adj": loss_rec_gae_adj.item()})
                if i % 10 == 0:
                    self.logger.info(f"Pretrain Stage 2: Epoch {i}, Loss: {loss.item()}")
                self.metrics.update_pretrain_loss(loss_rec_ae=loss_rec_ae.item(), loss_rec_gae=loss_rec_gae.item(), loss_rec_gae_adj=loss_rec_gae_adj.item())
        
        return z_tilde
        
    def clustering(self):
        X = self.graph.x
        A = self.graph.edge_index
        es_count = 0
        with torch.no_grad():
            # _, _, _, sim, _, _, _, Z, _, _ = self.forward(X, self.A_norm, X, self.A_norm)
            x, adj = self.graph.x, self.A_norm
            z_ae = self.ae.encoder(x)
            z_igae, sim, _, _ = self.gae.encoder(x, adj)
            z_i = self.a * z_ae + self.b * z_igae
            z_l = spmm(adj, z_i)
            s = torch.mm(z_l, z_l.t())
            s = F.softmax(s, dim=1)
            z_g = torch.mm(s, z_l)
            Z = self.alpha * z_g + z_l
            

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(Z.cpu().detach().numpy())
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        acc, nmi, ari, f1_macro, f1_weighted, _, _ = evaluate(y_pred, self.dataset.label)
        self.logger.info(f"Pretrain Scores: ACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
        
        Am = remove_edge(A, sim, remove_rate=0.1)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        
        with tqdm(range(self.epochs), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                X_tilde1, X_tilde2 = gaussian_noised_feature(X)
                
                # input & output
                X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = self.forward(X_tilde1, self.Ad, X_tilde2, Am)

                # calculate loss: L_{DICR}, L_{REC} and L_{KL}
                L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, self.gamma_value)
                L_REC = reconstruction_loss(X, self.A_norm, X_hat, Z_hat, A_hat)
                L_KL = distribution_loss(Q, target_distribution(Q[0].data))
                loss = L_DICR + L_REC + self.lambda_value * L_KL

                # optimization
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
                y_pred = KMeans(n_clusters=self.n_clusters, n_init=20).fit_predict(Z.data.cpu().numpy())
                
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, Z, y_true=self.graph.y)
                else:
                    _, (acc, nmi, ari, f1_macro, f1_weighted, _, _) = self.metrics.update(y_pred, y_true=self.graph.y)
                
                self.metrics.update_loss(total_loss=loss.item(), loss_dicr=L_DICR.item(), loss_rec=L_REC.item(), loss_kl=L_KL.item())
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tF1_macro: {f1_macro:.4f}\tF1_micro: {f1_weighted:.4f}")
                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "F1_macro": f1_macro,
                    "F1_weighted": f1_weighted,
                })
                
        return y_pred, Z
        
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):
        # node embedding encoded by AE
        Z_ae1 = self.ae.encoder(X_tilde1)
        Z_ae2 = self.ae.encoder(X_tilde2)

        # node embedding encoded by IGAE
        Z_igae1, A_igae1, AZ_1, Z_1 = self.gae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.gae.encoder(X_tilde2, Ad)

        # cluster-level embedding calculated by readout function
        Z_tilde_ae1 = self.R(Z_ae1)
        Z_tilde_ae2 = self.R(Z_ae2)
        Z_tilde_igae1 = self.R(Z_igae1)
        Z_tilde_igae2 = self.R(Z_igae2)

        # linear combination of view 1 and view 2
        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        # node embedding fusion from DFCN
        Z_i = self.a * Z_ae + self.b * Z_igae
        Z_l = spmm(Am, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # IGAE decoding
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.gae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + Z_adj_hat

        # node embedding and cluster-level embedding
        Z_ae_all = [Z_ae1, Z_ae2, Z_tilde_ae1, Z_tilde_ae2]
        Z_gae_all = [Z_igae1, Z_igae2, Z_tilde_igae1, Z_tilde_igae2]

        # the soft assignment distribution Q
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        # propagated embedding AZ_all and embedding Z_all
        AZ_en = []
        Z_en = []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i]+AZ_2[i])/2)
            Z_en.append((Z_1[i]+Z_2[i])/2)
        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all

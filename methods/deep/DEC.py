# Copyright (c) 2023 Yuxuan Shao

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

# This method reproduction refers to the following repository:
# https://github.com/piiswrong/dec/tree/master
# https://github.com/XifengGuo/IDEC
# https://github.com/vlukiyanov/pt-dec
from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from utils import config

from .base import DeepMethod
from .backbone.DEC_AE import DEC_AE
# from .backbone.EDESC_AE import EDESC_AE as DEC_AE
from .backbone.layers.DEC_ClusteringLayer import ClusteringLayer
from .utils.DEC_utils import target_distribution


class DEC(DeepMethod):
    def __init__(self, dataset, description, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.input_dim = dataset.input_dim
        self.encoder_dims = cfg.get("DEC", "encoder_dims")
        self.hidden_dim = cfg.get("DEC", "hidden_dim")
        self.alpha = cfg.get("DEC", "alpha")
        self.batch_size = cfg.get("DEC", "batch_size")
        self.pretrain_lr = cfg.get("DEC", "pretrain_learn_rate")
        self.lr = cfg.get("DEC", "learn_rate")
        self.momentum = cfg.get("DEC", "momentum")
        self.train_max_epoch = cfg.get("DEC", "train_max_epoch")
        self.weight_dir = cfg.get("global", "weight_dir")
        self.tol = cfg.get("DEC", "tol")

        self.ae = DEC_AE(self.input_dim, self.encoder_dims, self.hidden_dim).to(self.device)
        self.clustering_layer = ClusteringLayer(self.n_clusters, self.hidden_dim, self.alpha).to(self.device)
    
    def forward(self, x):
        x_bar, z = self.ae(x)
        q = self.clustering_layer(z)
        return x_bar, q, z
    
    def encode_dataset(self):
        self.eval()
        train_loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
        latent_list = []
        assign_list = []
        with torch.no_grad():
            for data, _, _ in tqdm(train_loader, desc="Encoding dataset", dynamic_ncols=True, leave=False):
                data = data.to(self.device)
                _, q, z = self(data)
                latent_list.append(z)
                assign_list.append(q)
        latent = torch.cat(latent_list, dim=0)
        assign = torch.cat(assign_list, dim=0)
        self.train()
        return latent, assign

    def pretrain(self):
        pretrain_path = self.cfg.get("DEC", "pretrain_file")
        if pretrain_path is not None:
            pretrain_path = os.path.join(self.weight_dir, pretrain_path)
        else:
            pretrain_path = ""
        if pretrain_path is not None and self.cfg.get("global", "use_pretrain") and os.path.exists(pretrain_path):
            self.logger.info(f"Pretrained weight found, Loading pretrained model in {pretrain_path}...")
            self.ae.load_state_dict(torch.load(pretrain_path))
        else:
            weight_path = os.path.join(self.weight_dir, f"{self.description}_pretrain.pth")
            if not os.path.exists(pretrain_path):
                self.logger.info("Pretrained weight not found, Pretraining...")
            elif not self.cfg.get("global", "use_pretrain"):
                self.logger.info("Not using pretrained weight, Pretraining...")
            
            if self.cfg.get("DEC", "layer_wise_pretrain"):
                # Pretrain in greedy layer-wise way
                train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
                with tqdm(range(len(self.encoder_dims) + 1), desc="Pretrain Stacked AE Period1", dynamic_ncols=True, leave=False) as level_loader:
                    for i in level_loader:
                        optimizer = optim.SGD(self.ae.parameters(), lr=self.pretrain_lr, momentum=self.momentum)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
                        with tqdm(range(50000), desc="Period1 Epoch", dynamic_ncols=True, leave=False) as epoch_loader:
                            for it in epoch_loader:
                                total_loss = 0
                                for data, _, _ in train_loader:
                                    data = data.to(self.device)
                                    x_bar, _ = self.ae(data, level=i)
                                    loss = nn.MSELoss()(x_bar, data)
                                    optimizer.zero_grad()
                                    loss.backward()
                                    total_loss += loss.item()
                                    optimizer.step()
                                scheduler.step()
                                self.metrics.update_pretrain_loss(total_loss=total_loss / len(train_loader))
                                epoch_loader.set_postfix_str(f"Loss {total_loss / len(train_loader):.4f}")
                                if it % 1000 == 0:
                                    self.logger.info(f"Pretrain Period1 Level {i} Epoch {it}\tLoss {total_loss / len(train_loader):.4f}")
                        self.ae.freeze_level(i)
                
                optimizer = optim.SGD(self.ae.parameters(), lr=self.pretrain_lr, momentum=self.momentum)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
                self.ae.defreeze()
                with tqdm(range(100000), desc="Pretrain Stacked AE Period2", dynamic_ncols=True, leave=False) as epoch_loader:
                    for it in epoch_loader:
                        total_loss = 0
                        for data, _, _ in train_loader:
                            data = data.to(self.device)
                            x_bar, _ = self.ae(data)
                            loss = nn.MSELoss()(x_bar, data)
                            total_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        scheduler.step()
                        self.metrics.update_pretrain_loss(total_loss=total_loss / len(train_loader))
                        epoch_loader.set_postfix_str(f"Loss {total_loss / len(train_loader):.4f}")
                        if it % 1000 == 0:
                            self.logger.info(f"Pretrain Period2 Epoch {it}\tLoss {total_loss / len(train_loader):.4f}")
            else:
                # Pretrain in a quick way (not layer-wise)
                optimizer = optim.Adam(self.ae.parameters(), lr = 0.001)
                with tqdm(range(100), desc="Pretrain Stacked AE Quickly", dynamic_ncols=True, leave=False) as epoch_loader:
                    for it in epoch_loader:
                        total_loss = 0
                        for data, _, _ in train_loader:
                            data = data.to(self.device)
                            x_bar, _ = self.ae(data)
                            loss = nn.MSELoss()(x_bar, data)
                            total_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        self.metrics.update_pretrain_loss(total_loss=total_loss / len(train_loader))
                        epoch_loader.set_postfix_str(f"Loss {total_loss / len(train_loader):.4f}")

            self.logger.info(f"Pretrain Weight saved in {weight_path}")
            torch.save(self.ae.state_dict(), weight_path)
            self.logger.info("Pretraining finished!")

        return self.encode_dataset()[0]

    def train_model(self):
        self.ae.defreeze()
        es_count = 0
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)

        z, _ = self.encode_dataset()
        y_pred= self.clustering_layer.kmeans_init(z)
        y_pred_last = y_pred
        with tqdm(range(self.train_max_epoch), desc="Clustering Training", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                total_loss = 0
                z, q = self.encode_dataset()
                p = target_distribution(q)
                y_pred = q.cpu().detach().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, z, y_true=self.dataset.label)
                else:
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, y_true=self.dataset.label)
                for data, _, idx in train_loader:
                    data = data.to(self.device)
                    x_bar, q, z = self(data)
                    loss = nn.KLDivLoss(reduction='batchmean')(q.log(), p[idx])
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss /= len(train_loader)
                self.metrics.update_loss(total_loss=total_loss)
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tDelta_label {delta_label:.4f}")
                if delta_label < self.tol and es_count > 5:
                    self.logger.info(f"Early stopping at epoch {epoch} with delta_label= {delta_label:.4f}")
                    break
                else:
                    es_count += 1
                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "Delta_label": delta_label
                })
        return y_pred, self.encode_dataset()[0]


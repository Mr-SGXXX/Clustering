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
# https://github.com/Yunfan-Li/Contrastive-Clustering/tree/main
from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from metrics import normalized_mutual_info_score as cal_nmi
from datasetLoader import ClusteringDataset
from utils import config
from .base import DeepMethod
from .backbone.CC_resnet import get_resnet, Network
from .loss.CC_loss import InstanceLoss, ClusterLoss
from .utils.CC_utils import Transforms


class CC(DeepMethod):
    def __init__(self, dataset: ClusteringDataset, description: str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.batch_size = self.cfg.get("CC", "batch_size")
        self.image_size = self.cfg.get("CC", "image_size")
        self.epochs = self.cfg.get("CC", "epochs")
        self.resume = self.cfg.get("CC", "resume")
        self.learn_rate = self.cfg.get("CC", "learn_rate")
        self.weight_decay = self.cfg.get("CC", "learn_rate")
        self.instance_temperature = self.cfg.get("CC", "instance_temperature")
        self.cluster_temperature = self.cfg.get("CC", "cluster_temperature")
        resnet = get_resnet(self.cfg.get("CC", "resnet"))
        self.model = Network(resnet, self.cfg.get(
            "CC", "feature_dim"), self.n_clusters).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=self.learn_rate, weight_decay=self.weight_decay)
        self.transform = Transforms(self.image_size)


    def encode_dataset(self):
        self.eval()
        train_loader = DataLoader(
            self.dataset, self.batch_size, shuffle=False, num_workers=self.workers)
        latent_list = []
        assign_list = []
        with torch.no_grad():
            for data, _, _ in tqdm(train_loader, desc="Encoding dataset", dynamic_ncols=True, leave=False):
                data = data.to(self.device)
                z, q = self(data)
                latent_list.append(z)
                assign_list.append(q)
        latent = torch.cat(latent_list, dim=0)
        assign = torch.cat(assign_list, dim=0)
        self.train()
        return latent, assign

    def pretrain(self):
        self.dataset.pretrain()
        return None

    def train_model(self):
        self.dataset.clustering()
        if self.resume is not None:
            checkpoint = torch.load(self.resume).to(self.device)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resuming from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            self.logger.info(
                f"Checkpoint not used or not founf, starting from scratch!")

        self.model.train()
        cluster_data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)
        if self.dataset.unlabel_data is not None:
            instance_data = TensorDataset(self.dataset.unlabel_data)
            instance_data_loader = DataLoader(
                instance_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)
        else:
            instance_data_loader = None
        criterion_instance = InstanceLoss(
            self.batch_size, self.instance_temperature, self.device)
        criterion_cluster = ClusterLoss(
            self.n_clusters, self.cluster_temperature, self.device)
        y_pred_last = np.zeros_like(self.dataset.label)
        with tqdm(range(self.start_epoch, self.epochs), desc="Training Epoch") as epoch_loader:
            for epoch in epoch_loader:
                total_instance_loss = 0
                total_cluster_loss = 0
                total_loss = 0
                if instance_data_loader is not None:
                    with tqdm(enumerate(instance_data_loader), desc="Training Batch Unlabeled") as batch_loader:
                        for i, batch in batch_loader:
                            self.optimizer.zero_grad()
                            batch = batch[0].to(self.device)
                            x_i, x_j = self.transform(batch)
                            z_i, z_j, c_i, c_j = self.model(x_i, x_j)
                            loss_instance = criterion_instance(z_i, z_j)
                            loss = loss_instance
                            loss.backward()
                            self.optimizer.step()
                            if i % 50 == 0:
                                self.logger.info(
                                    f"Epoch {epoch + 1} Batch {i + 1} Unlabeled Data: Instance Loss {loss_instance.item():.4f}")
                            batch_loader.set_postfix({
                                "Instance Loss": loss_instance,
                            })
                with tqdm(enumerate(cluster_data_loader), desc="Training Batch Labeled") as batch_loader:
                    for i, batch in batch_loader:
                        self.optimizer.zero_grad()
                        batch = batch[0].to(self.device)
                        x_i, x_j = self.transform(batch)
                        z_i, z_j, c_i, c_j = self.model(x_i, x_j)
                        loss_instance = criterion_instance(z_i, z_j)
                        loss_cluster = criterion_cluster(c_i, c_j)
                        loss = loss_instance + loss_cluster
                        loss.backward()
                        self.optimizer.step()
                        if i % 50 == 0:
                            self.logger.info(
                                f"Epoch {epoch + 1} Batch {i + 1} Labeled Data: Loss {loss.item():.4f} Instance Loss {loss_instance.item():.4f} Cluster Loss {loss_cluster.item():.4f}")
                        total_instance_loss += loss_instance.item()
                        total_cluster_loss += loss_cluster.item()
                        total_loss += loss.item()
                        batch_loader.set_postfix({
                            "Instance Loss": loss_instance,
                            "Cluster Loss": loss_cluster,
                            "Total Loss": loss
                        })
                self.metrics.update_pretrain_loss(
                    total_loss=total_loss,
                    instance_loss=total_instance_loss,
                    cluster_loss=total_cluster_loss
                )
                features, assign = self.encode_dataset()
                y_pred = assign.cpu().detach().numpy().argmax(1)
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, features, y_true=self.dataset.label)
                else:
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, y_true=self.dataset.label)
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch % 10 == 0:
                    self.logger.info(
                        f'Epoch {epoch + 1}\tAcc {acc:.4f}\tNMI {nmi:.4f}\tARI {ari:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}')

                epoch_loader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "Delta Label": delta_label,
                    "Delta NMI": delta_nmi
                })
        return y_pred, features


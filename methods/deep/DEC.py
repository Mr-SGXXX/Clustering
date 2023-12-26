from logging import Logger
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from metrics import Metrics
from utils import config

from .base import DeepMethod
from .backbone.DEC_AE import DEC_AE
from .backbone.layers.DEC_ClusteringLayer import ClusteringLayer
from .utils.DEC_utils import target_distribution
import torch
from torch.utils.data import DataLoader


class DEC(DeepMethod):
    def __init__(self, dataset, logger: Logger, cfg: config):
        super().__init__(dataset, logger, cfg)
        self.input_dim = dataset.input_dim
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            self.n_clusters = dataset.num_classes
        else:
            self.n_clusters = cfg.get("global", "n_clusters")
            assert type(
                self.n_clusters) is int, "n_clusters should be of type int"
            assert self.n_clusters > 0, "n_clusters should be larger than 0"
        self.encoder_dims = cfg.get("DEC", "encoder_dims")
        self.hidden_dim = cfg.get("DEC", "hidden_dim")
        self.alpha = cfg.get("DEC", "alpha")
        self.batch_size = cfg.get("DEC", "batch_size")
        self.pretrain_lr = cfg.get("DEC", "pretrain_learn_rate")
        self.lr = cfg.get("DEC", "learn_rate")
        self.train_max_epoch = cfg.get("DEC", "train_max_epoch")
        self.weight_dir = cfg.get("global", "weight_dir")

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
        weight_path = os.path.join(self.weight_dir, f"DEC_{self.dataset.name}_pretrain.pth")
        if os.path.exists(weight_path):
            self.logger.info("Pretrained weight found, Loading pretrained model...")
            self.ae.load_state_dict(torch.load(weight_path))
            pretrain_loss_list = []
        else:
            pretrain_loss_list = []
            self.logger.info("Pretrained weight not found, Pretraining...")
            optimizer = optim.Adam(self.ae.parameters(), lr=self.pretrain_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
            train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
            with tqdm(range(len(self.encoder_dims) + 1), desc="Pretrain Stacked AE Period1", dynamic_ncols=True, leave=False) as level_loader:
                for i in level_loader:
                    with tqdm(range(5000), desc="Period1 Epoch", dynamic_ncols=True, leave=False) as epoch_loader:
                        for it in epoch_loader:
                            total_loss = 0
                            for data, _, _ in train_loader:
                                data = data.to(self.device)
                                x_bar, _ = self.ae(data)
                                loss = nn.MSELoss()(x_bar, data)
                                optimizer.zero_grad()
                                loss.backward()
                                total_loss += loss.item()
                                optimizer.step()
                            scheduler.step()
                            pretrain_loss_list.append(total_loss / len(train_loader))
                            epoch_loader.set_postfix_str(f"Loss {total_loss / len(train_loader):.4f}")
                        
                    for j in range(len(self.ae.encoder)):
                        if j > i * 3:
                            break
                        self.ae.encoder[j].requires_grad_(False)
                        self.ae.decoder[-(j+1)].requires_grad_(False)

            
            optimizer = optim.Adam(self.ae.parameters(), lr=self.pretrain_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
            for param in self.ae.parameters():
                param.requires_grad_(True)
            for module in self.ae.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
            with tqdm(range(10000), desc="Pretrain Stacked AE Period2", dynamic_ncols=True, leave=False) as epoch_loader:
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
                    pretrain_loss_list.append(total_loss / len(train_loader))
                    epoch_loader.set_postfix_str(f"Loss {total_loss / len(train_loader):.4f}")
            torch.save(self.ae.state_dict(), weight_path)
            self.logger.info("Pretraining finished!")

        return self.encode_dataset()[0], pretrain_loss_list

    def train_model(self):
        metrics = Metrics(self.dataset.label is not None)
        optimizer = Adam(self.parameters(), lr=self.lr)
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
                _, (acc, nmi, ari, _, _) = metrics.update(y_pred, z.data.cpu().numpy(),
                                                          y_true=self.dataset.label)
                for data, _, idx in train_loader:
                    data = data.to(self.device)
                    x_bar, q, z = self(data)
                    loss = nn.KLDivLoss()(q.log(), p[idx])
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss /= len(train_loader)
                metrics.update_loss(total_loss=total_loss)
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}\tACC: {acc}\tNMI: {nmi}\tARI: {ari}\tDelta_label {delta_label:.4f}")
                if delta_label < 1e-5:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        return y_pred, self.encode_dataset()[0], metrics


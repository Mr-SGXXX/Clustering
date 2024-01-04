import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from logging import Logger
import numpy as np
from tqdm import tqdm

from metrics import Metrics
from utils import config
from .backbone.EDESC_AE import EDESC_AE
from .loss.EDESC_loss import D_constraint1, D_constraint2
from .utils.EDESC_utils import seperate, Initialization_D, refined_subspace_affinity
from .base import DeepMethod


class EDESC(DeepMethod):
    """
    EDESC model for deep subspace clustering.

    Args:
        dataset: Dataset object containing input dimension and label information.
        logger (Logger): Logger object for logging.
        cfg (config): Configuration object containing hyperparameters of the model.

    Attributes:
        input_dim (int): Input dimension. Automatically inferred from the dataset.
        dataset: Dataset object.
        n_clusters (int): Number of clusters.
        encoder_dims (list): List of encoder dimensions.
        decoder_dims (list): List of decoder dimensions.
        hidden_dim (int): Hidden layer dimension.
        d (int): Subspace dimension.
        eta (float): Subspace affinity parameter.
        beta (float): KL divergence weight parameter.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        logger (Logger): Logger object for logging.
        device (str): Device name.
        evaluate_scores (dict): Dictionary of evaluation scores.
        clustering_scores (dict): Dictionary of clustering evaluation scores.
        ae (EDESC_AE): EDESC autoencoder model.
        D (nn.Parameter): Proxy of subspace bases.

    """

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
        self.encoder_dims = cfg.get("EDESC", "encoder_dims")
        self.decoder_dims = cfg.get("EDESC", "decoder_dims")
        self.d = cfg.get("EDESC", "d")
        self.eta = cfg.get("EDESC", "eta")
        self.beta = cfg.get("EDESC", "beta")
        self.lr = cfg.get("EDESC", "learn_rate")
        self.batch_size = cfg.get("EDESC", "batch_size")

        self.evaluate_scores = {"acc": [], "nmi": [], "ari": []}
        self.clustering_scores = {"sc": []}

        # refer to github issue https://github.com/JinyuCai95/EDESC-pytorch/issues/1 which tells that hidden_dim(n_z) == d * n_clusters
        self.hidden_dim = self.d * self.n_clusters
        self.ae = EDESC_AE(self.input_dim, self.encoder_dims,
                           self.decoder_dims, self.hidden_dim).to(self.device)

        # Subspace bases proxy
        # TODO: I think the shape of the parameter should be (hidden_dim, n_clusters * d)
        self.D = Parameter(torch.Tensor(
            self.hidden_dim, self.n_clusters * self.d)).to(self.device)

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
        """
        Pretrain the EDESC autoencoder model.

        Returns:
            z (torch.Tensor): Hidden layer representation of whole dataset.
        """

        self.train()
        model = self.ae
        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = Adam(model.parameters(), lr=self.lr)
        pretrain_loss_list = []
        with tqdm(range(50), desc="Pretrain AE", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                total_loss = 0.
                for batch_idx, (x, _, _) in enumerate(train_loader):
                    x = x.to(self.device)
                    optimizer.zero_grad()
                    x_bar, _ = model(x)
                    loss = F.mse_loss(x_bar, x)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                if (epoch+1) % 10 == 0:
                    self.logger.info("Pretrain Epoch {} loss={:.4f}".format(
                        epoch + 1, total_loss / (batch_idx + 1)))
                epoch_loader.set_postfix({
                    "loss": total_loss / (batch_idx + 1)
                })
                pretrain_loss_list.append(total_loss / (batch_idx + 1))

        self.logger.info("Pretraining finished!")
        return self.encode_dataset()[0], pretrain_loss_list

    def train_model(self):
        """
        Train the EDESC model.

        Returns:
            y_pred (numpy.ndarray): Predicted cluster labels.
            z (torch.Tensor): Hidden layer representation.
            metrics (Metrics): Metrics object for evaluation.

        """
        # self.ae.load_state_dict(torch.load(
        #     "weight/reuters.pkl", map_location=self.device))
        metrics = Metrics(self.dataset.label is not None)
        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False)
        # d_cons1 = D_constraint1()
        # d_cons2 = D_constraint2()

        z, _ = self.encode_dataset()
        # data = torch.Tensor(self.dataset.data).to(self.device)
        # x_bar, z = self.ae(data)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        y_pred_last = y_pred
        D = Initialization_D(z, y_pred, self.n_clusters, self.d)
        self.D.data = torch.tensor(D).to(
            torch.float32).to(self.device)

        optimizer = Adam(self.parameters(), lr=self.lr)
        self.train()
        with tqdm(range(100), desc="Clustering", dynamic_ncols=True, leave=False) as epoch_loader:
            for epoch in epoch_loader:
                # _, tmp_s, z = self(data)
                z, tmp_s = self.encode_dataset()

                # Update refined subspace affinity
                tmp_s = tmp_s.data
                s_tilde = refined_subspace_affinity(tmp_s)

                # Evaluate clustering performance
                y_pred = tmp_s.cpu().detach().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                _, (acc, nmi, ari, _, _) = metrics.update(y_pred, z, y_true=self.dataset.label)
                if epoch % 10 == 0:
                    self.logger.info(
                        f'Epoch {epoch + 1}: Acc {acc:.4f} NMI {nmi:.4f} ARI {ari:.4f} Delta_label {delta_label:.4f}')
                total_reconstr_loss = 0
                total_kl_loss = 0
                total_loss_d1 = 0
                total_loss_d2 = 0
                total_loss = 0
                with tqdm(train_loader, desc="Epoch {}".format(epoch), dynamic_ncols=True, leave=False) as batch_loader:
                    for batch_idx, (x, _, idx) in enumerate(batch_loader):
                        x = x.to(self.device)
                        idx = idx.to(self.device)
                        x_bar, s, z = self(x)

                        # ~ y_pred = s.data.cpu().numpy().argmax(1).astype(np.float32)

                        ############# Total loss function ###################
                        # Reconstruction loss
                        reconstr_loss = F.mse_loss(x_bar, x)

                        # Subspace clustering loss
                        kl_loss = F.kl_div(
                            s.log(), s_tilde[idx])

                        # Constraints
                        d_cons1 = D_constraint1()
                        d_cons2 = D_constraint2()
                        loss_d1 = d_cons1(self.D)
                        loss_d2 = d_cons2(self.D, self.d, self.n_clusters)

                        # Total_loss
                        loss = reconstr_loss + self.beta * kl_loss + loss_d1 + loss_d2

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Update Loss Record
                        total_reconstr_loss += reconstr_loss.item()
                        total_kl_loss += kl_loss.item()
                        total_loss_d1 += loss_d1.item()
                        total_loss_d2 += loss_d2.item()
                        total_loss += loss.item()
                # Record Loss
                total_reconstr_loss /= len(train_loader)
                total_kl_loss /= len(train_loader)
                total_loss_d1 /= len(train_loader)
                total_loss_d2 /= len(train_loader)
                total_loss /= len(train_loader)
                metrics.update_loss(total_reconstr_loss=total_reconstr_loss,
                                    total_kl_loss=total_kl_loss,
                                    total_loss_d1=total_loss_d1,
                                    total_loss_d2=total_loss_d2,
                                    total_loss=total_loss)
                epoch_loader.set_postfix({
                    "Acc": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "Delta_label": delta_label
                })
        return y_pred, self.encode_dataset()[0], metrics

    def forward(self, x):
        """
        Forward pass function.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            x_bar (torch.Tensor): Reconstructed input data.
            s (torch.Tensor): Subspace affinity matrix.
            z (torch.Tensor): Hidden layer representation.

        """
        x_bar, z = self.ae(x)
        d = self.d
        s = None
        eta = self.eta

        # Calculate subspace affinity
        for i in range(self.n_clusters):

            si = torch.sum(
                torch.pow(torch.mm(z, self.D[:, i*d:(i+1)*d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z



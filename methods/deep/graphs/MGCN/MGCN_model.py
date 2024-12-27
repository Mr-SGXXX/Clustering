import torch
from torch import nn
# import torch.nn.functional as F
from .MGCN_AE import AE
from .MGCN_IGAE import IGAE

class MGCN_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, 
                 ae_n_dec_1, ae_n_dec_2, 
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, sigma, v=1.0):
        super(MGCN_model, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            n_input=n_input,
            n_z=n_z)

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

        self.s = nn.Sigmoid()
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v
        self.sigma = sigma 

    def init(self, X_tilde1, adj1):
        sigma = self.sigma
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
        z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
        z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
        z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1, active=False)
        z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3
        z_igae_adj = self.s(torch.mm(z_tilde, z_tilde.t()))

        x_hat = self.ae.decoder(z_ae3)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj1)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae3, z_igae3, q, q1, q2, z_tilde

    def forward(self, X_tilde1, adj1):
        sigma = self.sigma
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
        z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
        z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
        z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1, active=False)
        z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3
        z_igae_adj = self.s(torch.mm(z_igae3, z_igae3.t()))
        x_hat = self.ae.decoder(z_ae3)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj1)
        adj_hat = z_igae_adj + z_hat_adj
        q = 1.0 / (1.0 + torch.sum(torch.pow(z_tilde.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae3, z_igae3, q, q1, q2, z_tilde

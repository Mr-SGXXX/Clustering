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
import torch
import torch.nn as nn
from torch_geometric.utils import spmm

from ..DFCN.DFCN_AE import AE

# GNNLayer from DFCN
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # in the original implementation, the activation function is different for different datasets.
        # To make the implementation consistent and run on more dataset, we use the same activation function LeakyReLU for all datasets.
        # self.act = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = spmm(adj, support)
        az = spmm(adj, output)
        return output, az


# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]


# Improved Graph Auto Encoder from DFCN
class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # IGAE decoder
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)
        
    def forward(self, x, adj):
        z_igae, z_igae_adj, _, _ = self.encoder(x, adj)
        z_hat, z_hat_adj, _, _ = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat

# readout function
class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        # In the official implementation, the code is:
        Z_tilde = torch.cat(Z_tilde, dim=0)
        # the shape of this feature is (K * n_z), different from the paper description
        # but we dicide to keep the official implementation code
        return Z_tilde.view(1, -1)
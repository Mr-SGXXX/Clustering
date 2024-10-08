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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# The origianl implementation of GNNLayer in the SDCN paper.
# Replaced by GCNConv of PyG in this project, which is a more general implementation of GCN.
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

# class AE(nn.Module):

#     def __init__(self, input_dim, encoder_dims, hidden_dim):
#         super(AE, self).__init__()
#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
        
#         self.encoder_num = len(encoder_dims)
        
#         self.encoders.append(nn.Linear(input_dim, encoder_dims[0]))
#         self.decoders.append(nn.Linear(hidden_dim, encoder_dims[-1]))
#         for i in range(0, self.encoder_num-1):
#             self.encoders.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
#             self.decoders.append(nn.Linear(encoder_dims[-i-1], encoder_dims[-i-2]))
#         self.encoders.append(nn.Linear(encoder_dims[-1], hidden_dim))
#         self.decoders.append(nn.Linear(encoder_dims[0], input_dim))

#     def forward(self, x):
#         embeds = []
#         for i, encoder in enumerate(self.encoders):
#             x = encoder(x)
#             if i != self.encoder_num:
#                 x = F.relu(x)
#                 embeds.append(x)
#         z = x
#         for i, decoder in enumerate(self.decoders):
#             x = decoder(x)
#             if i != self.encoder_num:
#                 x = F.relu(x)

#         return x, z, embeds
    

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


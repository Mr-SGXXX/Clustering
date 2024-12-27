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
import torch.nn.functional as F


class GRACE_Model(nn.Module):
    def __init__(self, input_dim, encoder_dims, decoder_dims, embedding_dim, n_clusters, T, RI, RW, transition_function="RI", random_walk_step=2, dropout_rate=0.5, bn_flag=False, epsilon=1.0):
        super(GRACE_Model, self).__init__()
        
        if isinstance(encoder_dims, int):
            encoder_dims = [encoder_dims]
        if isinstance(decoder_dims, int):
            decoder_dims = [decoder_dims]
        
        
        self.encoder = nn.Sequential(
            *(
                [nn.Linear(input_dim, encoder_dims[0]), nn.ELU(), nn.Dropout(p=dropout_rate)] +
                [
                    layer
                    for i in range(len(encoder_dims) - 1)
                    for layer in [nn.Linear(encoder_dims[i], encoder_dims[i + 1]), nn.ELU(), nn.Dropout(p=dropout_rate)]
                ] +
                [nn.Linear(encoder_dims[-1], embedding_dim), nn.ELU(), nn.Dropout(p=dropout_rate)]
            )
        )

        self.decoder = nn.Sequential(
            *(
                [nn.Linear(embedding_dim, decoder_dims[0]), nn.ELU(), nn.Dropout(p=dropout_rate)] +
                [
                    layer
                    for i in range(len(decoder_dims) - 1)
                    for layer in [nn.Linear(decoder_dims[i], decoder_dims[i + 1]), nn.ELU(), nn.Dropout(p=dropout_rate)]
                ] +
                [nn.Linear(decoder_dims[-1], input_dim)]
            )
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        if bn_flag:
            self.bn = nn.BatchNorm1d(embedding_dim)
            
        self.n_clusters = n_clusters
        self.bn_flag = bn_flag
        self.epsilon = epsilon
        self.transition_function = transition_function
        self.random_walk_step = random_walk_step
        self.T = T
        self.RI = nn.Parameter(RI, requires_grad=False)
        self.RW = nn.Parameter(RW, requires_grad=False)
        
        
    def transform(self, Z):
        if self.transition_function == 'T':
            for _ in range(self.random_walk_step):
                Z = torch.sparse.mm(self.T, Z)
        elif self.transition_function == 'RI':
            Z = torch.matmul(self.RI.t(), Z)
        elif self.transition_function == 'RW':
            Z = torch.matmul(self.RW.t(), Z)
        else:
            raise ValueError('Invalid transition function')
        if self.bn_flag:
            Z = self.bn(Z)
        return Z

    def forward(self, X):
        Z = self.encoder(X)
        Z_transform = self.transform(Z)
        Z_t = Z_transform.unsqueeze(1).repeat(1, self.n_clusters, 1)
        Q = torch.pow(torch.sum((Z_t - self.cluster_layer) ** 2, dim=2) / self.epsilon + 1.0, -(self.epsilon + 1.0) / 2.0)
        Q = Q / Q.sum(dim=1, keepdim=True)  # Normalize along the second dimension
        X_hat = self.decoder(Z)
        return X_hat, Q, Z_transform

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
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F

from .MAGI_loss import Loss


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels, base_model, dropout: float = 0.5, ns: float = 0.5):
        super(Encoder, self).__init__()
        if base_model == "GraphSAGE":
            base_model = SAGEConv
        elif base_model == "GCN":
            base_model = GCNConv 
        self.dropout = dropout
        self.k = len(hidden_channels)
        self.ns = ns

        self.convs = nn.ModuleList()
        self.convs.extend([base_model(in_channels, hidden_channels[0])])

        for i in range(1, self.k):
            self.convs.extend(
                [base_model(hidden_channels[i-1], hidden_channels[i])])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.k):
            self.convs[i].reset_parameters()

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None, dropout=True):
        if not adjs:
            for i in range(self.k):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, edge_index)
                x = F.leaky_relu(x, self.ns)
        else:
            for i, (edge_index, _, size) in enumerate(adjs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = F.leaky_relu(x, self.ns)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, in_channels: int, project_hidden, activation=nn.PReLU, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.in_channels = in_channels
        self.project_hidden = project_hidden
        self.activation = activation
        self.Loss = Loss(temperature=self.tau)

        self.project = None
        if self.project_hidden is not None:
            self.project = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.project.extend(
                [nn.Linear(self.in_channels, self.project_hidden[0])])
            self.activations.extend([nn.PReLU(project_hidden[0])])
            for i in range(1, len(self.project_hidden)):
                self.project.extend(
                    [nn.Linear(self.project_hidden[i-1], self.project_hidden[i])])
                self.activations.extend([nn.PReLU(project_hidden[i])])

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None) -> torch.Tensor:
        x = self.encoder(x, edge_index, adjs)
        if self.project is not None:
            for i in range(len(self.project_hidden)):
                x = self.project[i](x)
                x = self.activations[i](x)
        return x

    def loss(self, x: torch.Tensor, mask):
        return self.Loss(x, mask)

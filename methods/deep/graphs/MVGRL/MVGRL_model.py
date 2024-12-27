# MIT License

# Copyright (c) 2018 Petar Veličković

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
from .MVGRL_layers import GCN, AvgReadout, Discriminator

class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        seq = seq.unsqueeze(0)
        adj = adj.unsqueeze(0)
        diff = diff.unsqueeze(0)
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()
    
    
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# this file is modified from https://github.com/JinyuCai95/EDESC-pytorch/blob/master/Constraint.py

class D_constraint1(torch.nn.Module):

    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).to(d.device)
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[1],d.shape[1]).to(d.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint



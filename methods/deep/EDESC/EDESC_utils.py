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

# This file is modified from https://github.com/JinyuCai95/EDESC-pytorch/blob/master/InitializeD.py
from collections import defaultdict
import numpy as np
import torch


def seperate(Z, y_pred, n_clusters):
    # n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    # Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                # Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([n_clusters * d, n_clusters * d])
    # U = np.zeros([Z.shape[1], n_clusters * d])
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    return D

def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()
 

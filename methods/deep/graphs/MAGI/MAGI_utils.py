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

import random
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

def get_sim(batch, adj, wt=20, wl=3):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()

    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    # src, dst = dict_r[idx[adj_batch.storage.row()[3].item()].item()], dict_r[idx[adj_batch.storage.col()[3].item()].item()]
    return batch, adj_batch

def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_

def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled
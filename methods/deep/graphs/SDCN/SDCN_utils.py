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
import scipy.sparse as sp
import torch
import torch_sparse
from torch_sparse import SparseTensor

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def normalize(mx: SparseTensor):
    """Row-normalize sparse matrix"""
    row, col, value = mx.coo()
    if value is None:
        value = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
    device = mx.device()
    # 将行和列索引组合成坐标矩阵
    indices = torch.stack([row, col], dim=0)

    mx = sp.coo_matrix((value.cpu().numpy(), indices.cpu().numpy()), shape=mx.sizes())
    
    mx = mx + mx.T.multiply(mx.T > mx) - mx.multiply(mx.T > mx)
    mx = mx + sp.eye(mx.shape[0])
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = sparse_mx_to_torch_sparse_tensor(mx).to(device)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    # mx = SparseTensor.to_symmetric(mx, reduce="max")
    # mx = mx + SparseTensor.eye(mx.size(0)).to_device(mx.device())
    
    # rowsum = torch_sparse.sum(mx, dim=1)
    # r_inv = rowsum.pow(-1).flatten()
    # r_inv[r_inv == float('inf')] = 0.
    # r_mat_inv = SparseTensor.eye(mx.size(0)).to_device(mx.device())
    # r_mat_inv = torch_sparse.set_diag(r_mat_inv, r_inv)
    # mx = r_mat_inv @ mx
    
    # mx = torch.sparse.FloatTensor(indices, value, mx.sizes())

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
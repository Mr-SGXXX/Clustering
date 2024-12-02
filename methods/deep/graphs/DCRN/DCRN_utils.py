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
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor
import scipy.sparse as sp
from tqdm import tqdm


def normalize_adj(adj:SparseTensor, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    adj_tmp = adj
    if not adj_tmp.has_value():
        adj_tmp = adj_tmp.fill_value(1.)
        
    # add the self_loop
    if self_loop:
        adj_tmp = torch_sparse.fill_diag(adj_tmp, 1)
    
    # calculate degree matrix and it's inverse matrix
    deg = torch_sparse.sum(adj_tmp, dim=1)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    
    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        deg_inv_sqrt = deg_inv.sqrt()
        norm_adj = torch_sparse.mul(adj_tmp, deg_inv_sqrt.view(-1, 1))
        norm_adj = torch_sparse.mul(norm_adj, deg_inv_sqrt.view(1, -1))
    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = torch_sparse.mul(adj_tmp, deg_inv.view(-1, 1))
    
    return norm_adj
    
    
def diffusion_adj(adj: SparseTensor, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self loop
    adj_tmp = torch_sparse.fill_diag(adj, 1)
    
    # calculate degree matrix and it's inverse matrix
    d = torch_sparse.sum(adj_tmp, dim=1)
    deg_inv_sqrt = d.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    
    # calculate the norm adj
    norm_adj = torch_sparse.mul(adj_tmp, deg_inv_sqrt.view(-1, 1))
    norm_adj = torch_sparse.mul(norm_adj, deg_inv_sqrt.view(1, -1))
    
    norm_adj = norm_adj.to_dense()
    if mode == "ppr":
        # diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))
        diff_adj = torch.inverse(torch.eye(d.shape[0]).to(norm_adj.device) - (1 - transport_rate) * norm_adj)
        diff_adj *= transport_rate
        diff_adj = SparseTensor.from_dense(diff_adj)
        
    return diff_adj


def remove_edge(A:SparseTensor, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    
    # remove edges based on cosine similarity of embedding
    n_node = A.size(0)
    A = A.to_dense()
    
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    A = SparseTensor.from_dense(A)
    
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    return Am

def gaussian_noised_feature(X:torch.Tensor):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(X.device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(X.device)
    X_tilde1 = X * N_1
    X_tilde2 = X * N_2
    return X_tilde1, X_tilde2

def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P
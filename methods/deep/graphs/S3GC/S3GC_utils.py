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
import torch_sparse
from torch.sparse import Tensor
from torch_geometric.utils import add_remaining_self_loops, is_torch_sparse_tensor, scatter, spmm, to_edge_index
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.sparse import set_sparse_value
import torch_geometric as pyg
import math


def ppr(i, alpha=0.2):
    return alpha*((1-alpha)**i)

def heat(i, t=5):
    return (math.e**(-t))*(t**i)/math.factorial(i)


def norm_adj(edge_index:Tensor, self_loop=True, num_nodes=None):
    adj_t = edge_index
    if self_loop:
        adj_t, _ = add_self_loops_fn(adj_t, None, 1, num_nodes)

    edge_index, value = to_edge_index(adj_t)
    col, row = edge_index[0], edge_index[1]

    deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

    return set_sparse_value(adj_t, value)

def compute_diffusion_matrix(adj:Tensor, x, niter=5, method="ppr"):
    print("Calculating S matrix")
    for i in range(0, niter):
        print("Iteration: " + str(i))
        if method=="ppr":
            theta = ppr(i)
        elif method=="heat":
            theta=heat(i)
        else:
            raise NotImplementedError
        if i==0:
            final = theta*x
            current = x
        else:
            current = torch.sparse.mm(adj, current)
            final+= (theta*current)
    return final
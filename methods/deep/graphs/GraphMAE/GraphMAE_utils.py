import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from functools import partial
import numpy as np

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph:Data, mask_prob):
    E = graph.num_edges

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(adj: SparseTensor,
              drop_rate: float,
              return_edges: bool = False):
    """
    在 torch_sparse.SparseTensor 表示的图上随机丢弃边。

    参数
    ----
    adj: torch_sparse.SparseTensor
        原始图的稀疏邻接矩阵。
    drop_rate: float
        丢弃边的比例（0 到 1）。若 <=0，则不丢弃任何边。
    return_edges: bool
        是否返回被丢弃的那部分边（以 SparseTensor 形式）。

    返回
    ----
    new_adj: torch_sparse.SparseTensor
        丢弃部分边并添加自环后的新图。
    dropped_adj: torch_sparse.SparseTensor（可选）
        仅当 return_edges=True 时返回，表示被丢弃的子图。
    """
    if drop_rate <= 0:
        if return_edges:
            empty = SparseTensor(row=torch.empty(0, dtype=torch.long, device=adj.storage.row().device),
                                 col=torch.empty(0, dtype=torch.long, device=adj.storage.row().device),
                                 sparse_sizes=adj.sizes())
            return adj, empty
        return adj

    # 提取 COO 格式的行、列、权重（三元组）
    row, col, value = adj.coo()
    num_edges = row.size(0)

    # 随机 mask，True 表示保留
    mask = torch.rand(num_edges, device=row.device) > drop_rate

    # 保留下来的边
    kept_row, kept_col = row[mask], col[mask]
    kept_val = value[mask] if value is not None else None

    # 丢弃的边
    dropped_row, dropped_col = row[~mask], col[~mask]
    dropped_val = value[~mask] if value is not None else None

    # 构造新的稀疏矩阵，并添加自环
    new_adj = SparseTensor(row=kept_row,
                           col=kept_col,
                           value=kept_val,
                           sparse_sizes=adj.sizes())
    new_adj = new_adj.set_diag()  # 在对角线上加自环

    if return_edges:
        dropped_adj = SparseTensor(row=dropped_row,
                                   col=dropped_col,
                                   value=dropped_val,
                                   sparse_sizes=adj.sizes())
        return new_adj, dropped_adj
    else:
        return new_adj


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

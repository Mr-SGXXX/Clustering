import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from .GraphMAE_utils import create_activation, create_norm

class GAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_hidden: int,
                 out_dim: int,
                 num_layers: int,
                 nhead: int,
                 nhead_out: int,
                 activation: str = 'relu',
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 norm=None,
                 concat_out: bool = False,
                 encoding: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.concat_out = concat_out
        self.encoding = encoding
        self.activation = create_activation(activation)
        self.feat_drop = nn.Dropout(feat_drop)

        self.last_activation = create_activation(activation) if encoding else None
        self.last_residual = residual if encoding else False
        self.last_norm = norm(out_dim * nhead_out) if (encoding and norm) else None

        # build layers
        self.convs = nn.ModuleList()
        self.res_conns = nn.ModuleList() if residual or self.last_residual else None
        self.norms = nn.ModuleList() if norm else None

        # single-layer
        if num_layers == 1:
            self._add_layer(in_dim, out_dim, nhead_out,
                            residual=self.last_residual,
                            norm_layer=self.last_norm,
                            concat=concat_out,
                            attn_drop=attn_drop,
                            negative_slope=negative_slope)
        else:
            self._add_layer(in_dim, num_hidden, nhead,
                            residual=residual,
                            norm_layer=norm(num_hidden * nhead) if norm else None,
                            concat=True,
                            attn_drop=attn_drop,
                            negative_slope=negative_slope)
            for _ in range(num_layers - 2):
                self._add_layer(num_hidden * nhead, num_hidden, nhead,
                                residual=residual,
                                norm_layer=norm(num_hidden * nhead) if norm else None,
                                concat=True,
                                attn_drop=attn_drop,
                                negative_slope=negative_slope)
            self._add_layer(num_hidden * nhead, out_dim, nhead_out,
                            residual=self.last_residual,
                            norm_layer=self.last_norm,
                            concat=concat_out,
                            attn_drop=attn_drop,
                            negative_slope=negative_slope)

        self.head = nn.Identity()

    def _add_layer(self, in_c, out_c, heads,
                   residual, norm_layer,
                   concat, attn_drop, negative_slope):
        # GATConv 自带 add_self_loops/bias
        conv = GATConv(in_c, out_c,
                       heads=heads,
                       concat=concat,
                       negative_slope=negative_slope,
                       dropout=attn_drop,
                       bias=True)
        self.convs.append(conv)

        if residual:
            if in_c != heads * out_c:
                self.res_conns.append(nn.Linear(in_c, heads * out_c, bias=False))
            else:
                self.res_conns.append(nn.Identity())

        if norm_layer is not None:
            self.norms.append(norm_layer)

    def forward(self, x, edge_index, return_hidden: bool = False):
        """
        x: Tensor [N, in_dim]
        edge_index: LongTensor [2, E]
        return_hidden: 是否返回每一层的中间表征
        """
        h = x
        hidden_states = []

        for l, conv in enumerate(self.convs):
            h_in = h
            h = self.feat_drop(h)
            h = conv(h, edge_index)  # [N, heads*out] 或者 [N, out]（concat=False）

            # residual
            if self.res_conns is not None:
                res = self.res_conns[l](h_in)
                h = h + res

            # norm
            if self.norms is not None:
                h = self.norms[l](h)

            # activation：末层只有 encoding=True 时才激活
            is_last = (l == self.num_layers - 1)
            if (not is_last) or (is_last and self.encoding):
                act = self.last_activation if (is_last and self.encoding) else self.activation
                if act is not None:
                    h = act(h)

            hidden_states.append(h)

        out = self.head(h)
        if return_hidden:
            return out, hidden_states
        else:
            return out

    def reset_classifier(self, num_classes: int):
        """
        将 head 换成线性分类器
        """
        self.head = nn.Linear(self.convs[-1].heads * self.convs[-1].out_channels,
                              num_classes)




class GINConv(MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 apply_func: nn.Module = None,
                 aggregator_type: str = "add",   # 对应 sum/mean/max
                 init_eps: float = 0.0,
                 learn_eps: bool = False,
                 residual: bool = False):
        super().__init__(aggr=aggregator_type)  # "add"/"mean"/"max"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.apply_func = apply_func

        # eps 参数
        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([init_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))

        # 残差分支
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, x, edge_index):
        """
        x: Tensor [N, in_dim]
        edge_index: LongTensor [2, E]
        """
        # propagate 会自动做 message + aggr -> out
        # out = \sum_j x_j  (或 mean/max)
        neigh = self.propagate(edge_index, x=x)  # [N, in_dim]

        # GIN 更新： (1 + eps) * x_i + neigh
        out = (1 + self.eps) * x + neigh  # [N, in_dim]

        # apply MLP+norm+act
        if self.apply_func is not None:
            out = self.apply_func(out)     # -> [N, out_dim]

        # 残差连接
        if self.res_fc is not None:
            out = out + self.res_fc(x)

        return out

    def message(self, x_j):
        # x_j: 邻居节点特征
        return x_j

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_dim}→{self.out_dim}, eps={float(self.eps):.4f})'


class ApplyNodeFunc(nn.Module):
    """对传入特征做 MLP → BN → Act"""
    def __init__(self, mlp: nn.Module, norm: str = "batchnorm", activation: str = "relu"):
        super().__init__()
        self.mlp = mlp
        norm_layer = create_norm(norm)
        self.norm = norm_layer(mlp.output_dim) if norm_layer else nn.Identity()
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """多层感知机，可选 BN+Act"""
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 activation: str = "relu",
                 norm: str = "batchnorm"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers == 1:
            # 线性模型
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            # 多层：Linear + Norm + Act
            self.linears = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.acts = nn.ModuleList()

            # 第一层
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            # 中间层
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            # 最后一层输出
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # Norm+Act 只在前 num_layers-1 层
            norm_layer = create_norm(norm)
            act_layer = create_activation(activation)
            for _ in range(num_layers - 1):
                self.norms.append(norm_layer(hidden_dim) if norm_layer else nn.Identity())
                self.acts.append(act_layer)

            self.linear_or_not = False

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            h = self.norms[i](h)
            h = self.acts[i](h)
        return self.linears[-1](h)


class GIN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_hidden: int,
                 out_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: str = "relu",
                 residual: bool = False,
                 norm: str = "batchnorm",
                 encoding: bool = False,
                 learn_eps: bool = False,
                 aggr: str = "add"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.out_dim = out_dim

        self.convs = nn.ModuleList()

        # 单层情形
        if num_layers == 1:
            mlp = MLP(2, in_dim, num_hidden, out_dim,
                      activation=activation, norm=norm)
            apply_func = ApplyNodeFunc(mlp, norm=norm, activation=activation) \
                         if (encoding and norm) else mlp
            conv = GINConv(in_dim, out_dim, apply_func,
                           aggregator_type=aggr, init_eps=0.,
                           learn_eps=learn_eps, residual=encoding and residual)
            self.convs.append(conv)
        else:
            # 输入投影
            mlp0 = MLP(2, in_dim, num_hidden, num_hidden,
                       activation=activation, norm=norm)
            self.convs.append(GINConv(in_dim, num_hidden,
                                      ApplyNodeFunc(mlp0, norm=norm, activation=activation),
                                      aggregator_type=aggr,
                                      init_eps=0., learn_eps=learn_eps,
                                      residual=residual))
            # 隐藏层
            for _ in range(num_layers - 2):
                mlp_h = MLP(2, num_hidden, num_hidden, num_hidden,
                            activation=activation, norm=norm)
                self.convs.append(GINConv(num_hidden, num_hidden,
                                          ApplyNodeFunc(mlp_h, norm=norm, activation=activation),
                                          aggregator_type=aggr,
                                          init_eps=0., learn_eps=learn_eps,
                                          residual=residual))
            # 输出投影
            mlp_out = MLP(2, num_hidden, num_hidden, out_dim,
                          activation=activation, norm=norm)
            apply_last = ApplyNodeFunc(mlp_out, norm=norm, activation=activation) \
                         if (encoding and norm) else mlp_out
            self.convs.append(GINConv(num_hidden, out_dim,
                                      apply_last,
                                      aggregator_type=aggr,
                                      init_eps=0., learn_eps=learn_eps,
                                      residual=encoding and residual))

        # 分类头
        self.head = nn.Identity()

    def forward(self, x, edge_index, return_hidden: bool=False):
        h = x
        hidden_list = []
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            hidden_list.append(h)
        out = self.head(h)
        if return_hidden:
            return out, hidden_list
        else:
            return out

    def reset_classifier(self, num_classes: int):
        self.head = nn.Linear(self.out_dim, num_classes)

class GraphConv(MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 norm=None,                # 传入归一化层类（如 nn.BatchNorm1d）或 None
                 activation=None,          # 传入激活函数实例，如 create_activation("relu")
                 residual: bool = True):
        super().__init__(aggr='add')  # sum 聚合
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 线性映射
        self.fc = nn.Linear(in_dim, out_dim)

        # 残差分支
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # 归一化层（BatchNorm1d 或 LayerNorm）
        self.norm = norm(out_dim) if norm is not None else None
        # 激活
        self.act = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        if isinstance(self.res_fc, nn.Linear):
            self.res_fc.reset_parameters()
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        """
        N = x.size(0)
        row, col = edge_index

        # 1) 源节点 out-deg^{-1/2}
        deg_out = degree(row, N, dtype=x.dtype).clamp(min=1)
        norm_out = deg_out.pow(-0.5).unsqueeze(-1)  # [N,1]
        x_src = x * norm_out

        # 2) 消息传递与求和
        out = self.propagate(edge_index, x=x_src)  # [N, in_dim]

        # 3) 线性变换
        out = self.fc(out)

        # 4) 目标节点 in-deg^{-1/2}
        deg_in = degree(col, N, dtype=x.dtype).clamp(min=1)
        norm_in = deg_in.pow(-0.5).unsqueeze(-1)
        out = out * norm_in

        # 5) 残差
        if self.res_fc is not None:
            out = out + self.res_fc(x)

        # 6) 归一化 + 激活
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out

    def message(self, x_j):
        # 直接把源节点特征作为消息
        return x_j

class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_hidden: int,
                 out_dim: int,
                 num_layers: int,
                 dropout: float,
                 activation: str,
                 residual: bool,
                 norm=None,            # 传入规范化层类，或 None
                 encoding: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim

        # 激活工厂
        act_fn = create_activation(activation)
        # 只有在 encoding=True 时，最后一层才加激活/归一化
        last_act = act_fn if encoding else None
        last_norm = norm if encoding else None

        self.convs = nn.ModuleList()
        if num_layers == 1:
            # 单层
            self.convs.append(
                GraphConv(in_dim, out_dim,
                          norm=last_norm,
                          activation=last_act,
                          residual=encoding and residual)
            )
        else:
            # 输入投影层（带激活/归一化）
            self.convs.append(
                GraphConv(in_dim, num_hidden,
                          norm=norm,
                          activation=act_fn,
                          residual=residual)
            )
            # 隐藏层
            for _ in range(num_layers - 2):
                self.convs.append(
                    GraphConv(num_hidden, num_hidden,
                              norm=norm,
                              activation=act_fn,
                              residual=residual)
                )
            # 输出投影层
            self.convs.append(
                GraphConv(num_hidden, out_dim,
                          norm=last_norm,
                          activation=last_act,
                          residual=encoding and residual)
            )

        # 分类头 or 占位 Identity
        self.head = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                return_hidden: bool = False):
        h = x
        hidden_list = []
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            hidden_list.append(h)
        out = self.head(h)
        if return_hidden:
            return out, hidden_list
        else:
            return out

    def reset_classifier(self, num_classes: int):
        self.head = nn.Linear(self.out_dim, num_classes)

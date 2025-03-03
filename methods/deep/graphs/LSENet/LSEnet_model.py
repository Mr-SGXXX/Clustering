import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .LSEnet_layers import LSENetLayer
from .LSEnet_lorentz import Lorentz
from .LSEnet_encoders import GraphEncoder
from .LSEnet_utils import select_activation


MIN_NORM = 1e-15
EPS = 1e-6

class LSENet_model(nn.Module):
    def __init__(self, manifold, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None):
        super(LSENet_model, self).__init__()
        if max_nums is not None:
            assert len(max_nums) == height - 1, "length of max_nums must equal height-1."
        self.manifold:Lorentz = manifold
        self.nonlin = select_activation(nonlin) if nonlin is not None else None
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.height = height
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = GraphEncoder(self.manifold, 2, in_features + 1, hidden_dim_enc, embed_dim + 1, use_att=False,
                                                     use_bias=True, dropout=dropout, nonlin=self.nonlin)
        self.layers = nn.ModuleList([])
        if max_nums is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            max_nums = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        for i in range(height - 1):
            self.layers.append(LSENetLayer(self.manifold, embed_dim + 1, hidden_features, max_nums[i],
                                           bias=True, use_att=False, dropout=dropout,
                                           nonlin=self.nonlin, temperature=self.temperature))
            # self.num_max = int(self.num_max / decay_rate)

    def forward(self, x, edge_index):

        """mapping x into Lorentz model"""
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        z = self.embed_layer(x, edge_index)
        z = self.normalize(z)

        self.tree_node_coords = {self.height: z}
        self.assignments = {}

        edge = edge_index.clone()
        ass = None
        for i, layer in enumerate(self.layers):
            z, edge, ass = layer(z, edge)
            self.tree_node_coords[self.height - i - 1] = z
            self.assignments[self.height - i] = ass

        self.tree_node_coords[0] = self.manifold.Frechet_mean(z)
        self.assignments[1] = torch.ones(ass.shape[-1], 1).to(x.device)

        return self.tree_node_coords, self.assignments

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x

class HyperSE(nn.Module):
    def __init__(self, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None):
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = Lorentz()
        self.encoder = LSENet_model(self.manifold, in_features, hidden_dim_enc, hidden_features,
                              num_nodes, height, temperature, embed_dim, dropout,
                              nonlin, decay_rate, max_nums)

    def forward(self, data, device=torch.device('cuda:0')):
        features = data['feature'].to(device)
        adj = data['adj'].to(device)
        embeddings, clu_mat = self.encoder(features, adj)
        self.embeddings = {}
        for height, x in embeddings.items():
            self.embeddings[height] = x.detach()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
        for k, v in ass_mat.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            ass_mat[k] = t
        self.ass_mat = ass_mat
        return self.embeddings[self.height]

    def loss(self, data, edge_index, neg_edge_index, device=torch.device('cuda:0'), pretrain=False):
        """_summary_

        Args:
            data: dict
            device: torch.Device
        """
        weight = data['weight']
        adj = data['adj'].to(device)
        degrees = data['degrees']
        features = data['feature']

        embeddings, clu_mat = self.encoder(features, adj)

        se_loss = 0
        vol_G = weight.sum()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
            vol_dict[k] = torch.einsum('ij, i->j', ass_mat[k], degrees)

        edges = torch.concat([edge_index, neg_edge_index], dim=-1)
        prob = self.manifold.dist(embeddings[self.height][edges[0]], embeddings[self.height][edges[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.concat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(device)
        lp_loss = F.binary_cross_entropy(prob, label)

        if pretrain:
            return self.manifold.dist0(embeddings[0]) + lp_loss

        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', clu_mat[k], vol_dict[k - 1])  # (N_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (N_k, )
            ass_i = ass_mat[k][edge_index[0]]   # (E, N_k)
            ass_j = ass_mat[k][edge_index[1]]
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # (N_k, )
            delta_vol = vol_dict[k] - weight_sum    # (N_k, )
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss + self.manifold.dist0(embeddings[0]) + lp_loss
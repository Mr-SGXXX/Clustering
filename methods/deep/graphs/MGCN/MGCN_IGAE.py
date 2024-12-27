import torch
from torch import nn
from torch.nn import Module, Parameter

class NGNNLayer(Module):
    def __init__(self, in_features, out_features, reduction=4, order=4):
        super(NGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.main_layer = [NGNN(self.in_features, self.out_features, i) for i
                            in range(1, self.order + 1)]
        self.main_layers = torch.nn.ModuleList(self.main_layer)

        self.fc1 = nn.Linear(out_features, out_features // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features // reduction, out_features)

        self.softmax = nn.Softmax(dim=0)
    def forward(self, features, adj, active=True):

        abstract_features = [self.main_layers[i](features, adj, active=active) for i in range(self.order)]
        feats_mean = [torch.mean(abstract_features[i], 0, keepdim=True) for i in range(self.order)]
        feats_mean = torch.cat(feats_mean, dim=0)
        feats_a = self.fc2(self.relu(self.fc1(feats_mean)))
        feats_a = self.softmax(feats_a)

        feats = []
        for i in range(self.order):
            feats.append(abstract_features[i] * feats_a[i])

        output = feats[0]
        for i in range(1, self.order):
            output += feats[i]

        return output

class NGNN(Module):
    def __init__(self, in_features, out_features, order=1):
        super(NGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = nn.Tanh()        #elu, prelu
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        output = torch.mm(features, self.weight)
        if active:
            output = self.act(output)

        output = torch.spmm(adj, output)
        for _ in range(self.order-1):
            output = torch.spmm(adj, output)
        return output

class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = NGNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = NGNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = NGNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj)
        z = self.gnn_2(z, adj)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = NGNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = NGNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = NGNNLayer(gae_n_dec_3, n_input)

        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj)
        z = self.gnn_5(z, adj)
        z_hat = self.gnn_6(z, adj, active=False)

        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj

class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat

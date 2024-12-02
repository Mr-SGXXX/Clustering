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
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import h5py

from ..layers import build_MLP_net

class DEC_AE(nn.Module):
    def __init__(self, n_input, encoder_dims, n_z):
        super(DEC_AE, self).__init__()
        encoder_dims = [n_input] + encoder_dims + [n_z]
        autoencoders = []
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
        for i in range(0, len(encoder_dims)-1):
            autoencoders.append(AutoEncoder(encoder_dims[i], encoder_dims[i+1]))
        self.autoencoders = nn.ModuleList(autoencoders)
    
    def forward(self, x, level:int=None):
        z = x
        if level is None:
            level = len(self.autoencoders)
        else:
            level += 1
        for i in range(level):
            z = self.autoencoders[i].encoder(z)
            if i != level-1:
                z = self.act(z)
                z = self.dropout(z)
        x_bar = z
        for i in range(level-1, -1, -1):
            x_bar = self.autoencoders[i].decoder(x_bar)
            if i != 0:
                x_bar = self.act(x_bar)
                x_bar = self.dropout(x_bar)
        return x_bar, z
    
    def freeze_level(self, level):
        for param in self.autoencoders[level].parameters():
            param.requires_grad = False

    def defreeze(self):
        self.dropout.p = 0.0
        for param in self.parameters():
            param.requires_grad = True

    def load_keras_weight(self, weight_path):
        weight = {}
        bias = {}
        with h5py.File(weight_path, 'r') as f:
            for layer in f.keys():
                if layer != "input":
                    weight[layer] = np.array(f[f"{layer}/{layer}/kernel:0"])
                    bias[layer] = np.array(f[f"{layer}/{layer}/bias:0"])
        device = self.autoencoders[0].encoder.weight.device
        self.autoencoders[0].encoder.weight.data = torch.from_numpy(weight['encoder_1'].T).float().to(device)
        self.autoencoders[0].encoder.bias.data = torch.from_numpy(bias['encoder_1']).float().to(device)
        self.autoencoders[0].decoder.weight.data = torch.from_numpy(weight['output'].T).float().to(device)
        self.autoencoders[0].decoder.bias.data = torch.from_numpy(bias['output']).float().to(device)
        self.autoencoders[1].encoder.weight.data = torch.from_numpy(weight['encoder_2'].T).float().to(device)
        self.autoencoders[1].encoder.bias.data = torch.from_numpy(bias['encoder_2']).float().to(device)
        self.autoencoders[1].decoder.weight.data = torch.from_numpy(weight['decoder_1'].T).float().to(device)
        self.autoencoders[1].decoder.bias.data = torch.from_numpy(bias['decoder_1']).float().to(device)
        self.autoencoders[2].encoder.weight.data = torch.from_numpy(weight['encoder_3'].T).float().to(device)
        self.autoencoders[2].encoder.bias.data = torch.from_numpy(bias['encoder_3']).float().to(device)
        self.autoencoders[2].decoder.weight.data = torch.from_numpy(weight['decoder_2'].T).float().to(device)
        self.autoencoders[2].decoder.bias.data = torch.from_numpy(bias['decoder_2']).float().to(device)
        self.autoencoders[3].encoder.weight.data = torch.from_numpy(weight['hidden'].T).float().to(device)
        self.autoencoders[3].encoder.bias.data = torch.from_numpy(bias['hidden']).float().to(device)
        self.autoencoders[3].decoder.weight.data = torch.from_numpy(weight['decoder_3'].T).float().to(device)
        self.autoencoders[3].decoder.bias.data = torch.from_numpy(bias['decoder_3']).float().to(device)

        

            
class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_z):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(n_input, n_z)
        self.decoder = nn.Linear(n_z, n_input)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x):
        h = self.encoder(x)
        x_bar = self.decoder(h)
        return x_bar, h

def init_weights(m):
    if type(m) == nn.Linear:
        init.normal_(m.weight, mean=0, std=0.01)
        init.constant_(m.bias, 0)

# class DEC_AE(nn.Module):
#     def __init__(self, n_input, encoder_dims, n_z):
#         super(DEC_AE, self).__init__()
#         encoder_dims = [n_input] + encoder_dims + [n_z]
#         self.encoder, self.decoder = create_layers(len(encoder_dims), encoder_dims, 0.2)
    
#     def forward(self, x):
#         z = self.encoder(x)
#         x_bar = self.decoder(z)
#         return x_bar, z

# def create_layers(n_layer, dims, drop):
#     encoder_layers = []
#     decoder_layers = []

#     for i in range(0, n_layer-1):
#         # encoder
#         encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
#         if i != n_layer - 2:
#             encoder_layers.append(nn.ReLU())
#             encoder_layers.append(nn.Dropout(drop))
#     for i in range(n_layer-2, 0, -1):
#         # decoder
#         if i != 0:
#             decoder_layers.append(nn.Linear(dims[i+1], dims[i]))
#             decoder_layers.append(nn.ReLU())
#             decoder_layers.append(nn.Dropout(drop))

#     # No ReLU and Dropout for the last layer
#     decoder_layers.append(nn.Linear(dims[1], dims[0]))

#     return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)
import torch.nn as nn
import torch

from layers import build_MLP_net

class EDESC_AE(nn.Module):
    def __init__(self, n_input, encoder_dims, decoder_dims, n_z):
        super(EDESC_AE, self).__init__()
        self.encoder = build_MLP_net([n_input] + encoder_dims + [n_z])
        self.decoder = build_MLP_net([n_z] + decoder_dims + [n_input])
    
    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        return x_bar, z

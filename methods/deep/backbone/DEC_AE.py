import torch
import torch.nn as nn

from .layers import build_MLP_net


class DEC_AE(nn.Module):
    def __init__(self, n_input, encoder_dims, n_z):
        super(DEC_AE, self).__init__()
        encoder_dims = [n_input] + encoder_dims + [n_z]
        self.encoder, self.decoder = create_layers(len(encoder_dims), encoder_dims, 0.2)
    
    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        return x_bar, z

def create_layers(n_layer, dims, drop):
    encoder_layers = []
    decoder_layers = []

    for i in range(0, n_layer-1):
        # encoder
        encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
        if i != n_layer - 2:
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(drop))
    for i in range(n_layer-2, 0, -1):
        # decoder
        if i != 0:
            decoder_layers.append(nn.Linear(dims[i+1], dims[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(drop))

    # No ReLU and Dropout for the last layer
    decoder_layers.append(nn.Linear(dims[1], dims[0]))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)
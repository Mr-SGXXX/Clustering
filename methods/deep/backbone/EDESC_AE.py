import torch.nn as nn
import torch.nn.functional as F

from .layers import build_MLP_net

# class EDESC_AE(nn.Module):
#     def __init__(self, n_input, encoder_dims, decoder_dims, n_z):
#         super(EDESC_AE, self).__init__()
#         self.encoder = build_MLP_net([n_input] + encoder_dims + [n_z])
#         self.decoder = build_MLP_net([n_z] + decoder_dims + [n_input])
    
#     def forward(self, x):
#         z = self.encoder(x)
#         x_bar = self.decoder(z)
#         return x_bar, z
    

class EDESC_AE(nn.Module):

    def __init__(self, n_input, encoder_dims, decoder_dims, n_z):
        super(EDESC_AE, self).__init__()

        # Encoder
        self.enc_1 = nn.Linear(n_input, encoder_dims[0])
        self.enc_2 = nn.Linear(encoder_dims[0], encoder_dims[1])
        self.enc_3 = nn.Linear(encoder_dims[1], encoder_dims[2])

        self.z_layer = nn.Linear(encoder_dims[2], n_z)

        # Decoder
        self.dec_1 = nn.Linear(n_z, decoder_dims[0])
        self.dec_2 = nn.Linear(decoder_dims[0], decoder_dims[1])
        self.dec_3 = nn.Linear(decoder_dims[1], decoder_dims[2])

        self.x_bar_layer = nn.Linear(decoder_dims[2], n_input)

    def forward(self, x):

        # Encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # Decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z
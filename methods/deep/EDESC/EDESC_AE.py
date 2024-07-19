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

import torch.nn as nn
import torch.nn.functional as F

from ..layers import build_MLP_net

# This file is modified from https://github.com/JinyuCai95/EDESC-pytorch/blob/master/AutoEncoder.py

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
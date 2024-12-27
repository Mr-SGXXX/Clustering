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

from torch import nn
from torch.nn import Linear


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.z_layer = Linear(ae_n_enc_2, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z1 = self.act(self.enc_1(x))
        z2 = self.act(self.enc_2(z1))
        z_ae = self.z_layer(z2)
        return z1, z2, z_ae


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.x_bar_layer = Linear(ae_n_dec_2, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            n_input=n_input,
            n_z=n_z)

    def forward(self, x):
        z1, z2, z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return z1, z2, z_ae, x_hat
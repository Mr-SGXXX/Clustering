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

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

def build_MLP_net(net_dims:list, activation=nn.ReLU, last_act=False, noise_sigma=0.0, last_noise=False, add_bn=False, drop_out:float=0.0):
    """
    Build a MLP network with given dimensions.
    
    Args:
        net_dims: A list of dimensions of each layer.
        activation: Activation function.
        last_act: Whether to add activation function to the last layer.
        noise_sigma: Add Gaussian noise to the output of each layer.
        last_noise: Whether to add Gaussian noise to the output of the last layer.
        add_bn: Whether to add batch normalization to the output of each layer.
        drop_out: Dropout rate.
    
    Returns:
    if noise_sigma != 0:
        Clean and Noise nn.Sequential object which shares weights.
    else:
        Clean nn.Sequential object.
    """
    clean_net, noise_net = [], []
    if noise_sigma != 0:
        noise_net.append(AddGaussianNoise(noise_sigma))
    for i in range(1, len(net_dims)):
        linear = nn.Linear(net_dims[i - 1], net_dims[i])
        clean_net.append(linear)
        if noise_sigma != 0:
            noise_net.append(linear)
            if last_noise or i != len(net_dims) - 1:
                noise_net.append(AddGaussianNoise(noise_sigma))
        if last_act or i != len(net_dims) - 1:
            if add_bn:
                clean_net.append(nn.BatchNorm1d(net_dims[i])) 
                if noise_sigma != 0:
                    noise_net.append(nn.BatchNorm1d(net_dims[i]))
            act = activation()
            clean_net.append(act)
            if noise_sigma != 0:
                noise_net.append(act)
            if drop_out != 0.0:
                clean_net.append(nn.Dropout1d(drop_out))
                if noise_sigma != 0:
                    noise_net.append(nn.Dropout1d(drop_out))
    if noise_sigma != 0:
        return nn.Sequential(*clean_net), nn.Sequential(*noise_net)
    else:
        return nn.Sequential(*clean_net)
    

class AddGaussianNoise(nn.Module):
    def __init__(self, sigma=0.0):
        super(AddGaussianNoise, self).__init__()
        self.sigma = sigma
        
    def forward(self, x):
        if self.training and self.sigma != 0:
            return x + torch.randn_like(x).to(x.device) * self.sigma
        else:
            return x
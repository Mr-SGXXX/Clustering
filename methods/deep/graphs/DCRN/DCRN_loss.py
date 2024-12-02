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
import torch.nn.functional as F
from torch_geometric.utils import spmm

def cross_correlation(Z_v1, Z_v2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())

def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def dicr_loss(Z_ae, Z_igae, AZ, Z, gamma_value):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
        gamma_value: the hyper-parameter
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    # loss of SCR
    L_N_ae = correlation_reduction_loss(S_N_ae)
    L_N_igae = correlation_reduction_loss(S_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)
    L_F_igae = correlation_reduction_loss(S_F_igae)

    L_N = 0.1 * L_N_ae + 5 * L_N_igae
    L_F = L_F_ae + L_F_igae

    # propagated regularization
    L_R = r_loss(AZ, Z)

    # loss of DICR
    loss_dicr = L_N + L_F + gamma_value * L_R

    return loss_dicr

# clustering guidance from DFCN
def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss

# the reconstruction function from DFCN
def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """
    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, spmm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    loss_igae = loss_w + 0.1 * loss_a
    loss_rec = loss_ae + loss_igae
    return loss_rec

def r_loss(AZ, Z):
    """
    the loss of propagated regularization (L_R)
    Args:
        AZ: the propagated embedding
        Z: embedding
    Returns: L_R
    """
    loss = 0
    for i in range(2):
        for j in range(3):
            p_output = F.softmax(AZ[i][j], dim=1)
            q_output = F.softmax(Z[i][j], dim=1)
            log_mean_output = ((p_output + q_output) / 2).log()
            loss += (F.kl_div(log_mean_output, p_output, reduction='batchmean') +
                     F.kl_div(log_mean_output, p_output, reduction='batchmean')) / 2
    return loss
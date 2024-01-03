from collections import defaultdict
import numpy as np
import torch


def seperate(Z, y_pred, n_clusters):
    # n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    # Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                # Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    Z_seperate = seperate(Z, y_pred, n_clusters)
    # U = np.zeros([n_clusters * d, n_clusters * d])
    U = np.zeros([Z.shape[1], n_clusters * d])
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    return D

def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()
 

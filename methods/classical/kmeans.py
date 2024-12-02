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
import numpy as np
from logging import Logger
from sklearn.cluster import KMeans as skKMeans
from torch_kmeans import KMeans as cudaKMeans
import torch
from functools import partial
from tqdm import tqdm

from utils import config
from datasetLoader import ClusteringDataset

from .base import ClassicalMethod

class KMeans(ClassicalMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.max_iterations = cfg.get("KMeans", "max_iterations")
        self.batch_size = cfg.get("KMeans", "batch_size")

    def clustering(self):
        if self.batch_size == -1:
            if self.device.startswith("cuda"):
                rst = cudaKMeans(n_clusters=self.n_clusters, max_iter=self.max_iterations, seed=self.cfg["global"]["seed"]).fit_predict(torch.tensor(self.dataset.data, device=self.device).unsqueeze(0))
                return rst.view(-1).cpu().numpy(), self.dataset.data
            else:
                return skKMeans(n_clusters=self.n_clusters, max_iter=self.max_iterations, random_state=self.cfg["global"]["seed"]).fit_predict(self.dataset.data), self.dataset.data
        else:
            rst, _ = batch_kmeans(torch.tensor(self.dataset.data, device=self.device), self.n_clusters, batch_size=self.batch_size, iter_limit=self.max_iterations, seed=self.cfg["global"]["seed"], device=self.device)
            return rst.cpu().detach().numpy(), self.dataset.data
        # return kmeans(self.dataset.data, self.n_clusters, self.max_iterations)

# these are my implementation of kmeans, not used in this project
def kmeans(data, k, max_iterations=100, init='kmeans++'):
    """
    K-means clustering algorithm.
    
    Args:
        data (array-like): Input data matrix of shape (n_samples, n_features).
        k (int): The number of clusters to form.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 100.
    """
    if init == 'kmeans++':
        centroids = kmeans_plusplus(data, k)
    elif init == 'random':
        centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    for _ in range(max_iterations):
        # Assign data points to the nearest cluster center
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1), axis=-1)
        
        # Update cluster centers to the mean of each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Stop iteration if cluster centers no longer change
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels

def kmeans_plusplus(data, k):
    """
    K-means++ initialization.
    
    Args:
        data (array-like): Input data matrix of shape (n_samples, n_features).
        k (int): The number of clusters to form.
    """
    # Randomly select the first centroid from the data
    centroids = [data[np.random.choice(range(len(data)))]]
    
    for _ in range(1, k):
        # Compute the distance from each data point to the nearest centroid
        dist = np.min([np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0)
        
        # Select a new centroid with probability proportional to dist
        probs = dist / np.sum(dist)
        centroids.append(data[np.random.choice(range(len(data)), p=probs)])
    
    return np.array(centroids)


# This method reproduction refers to the following repository:
# https://github.com/EdisonLeeeee/MAGI
def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def batch_kmeans(
        X,
        num_clusters,
        distance='euclidean',
        batch_size=100000,
        cluster_centers=[],
        tol=1e-4,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running minibatch k-means on {device}..')


    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    if type(cluster_centers) == list:
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0

    with tqdm(desc='[running kmeans]', dynamic_ncols=True, leave=False) as tqdm_meter:
        while True:
            choice_cluster = pairwise_distance_function(X, initial_state)

            initial_state_pre = initial_state.clone()

            for index in range(num_clusters):
                # selected = idx[choice_cluster == index].to(device)
                # selected = X[selected]
                selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
                selected = torch.index_select(X, 0, selected)


                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]

                initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1

            # update tqdm meter
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break

    return choice_cluster.cpu(), initial_state.cpu()


def batch_kmeans_predict(
        X,
        cluster_centers,
        batch_size=100000,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=True
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    choice_cluster = pairwise_distance_function(X, cluster_centers, batch_size=batch_size)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, batch_size=100000, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)
    if batch_size == -1:
        # full batch kmeans
        dis_ = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis_ = dis_.sum(dim=-1).squeeze()
        return torch.argmin(dis_, dim=1)
    else:
        # mini-batch kmeans
        choice_cluster = torch.zeros(data1.shape[0])
        for batch_idx in tqdm(range(int(np.ceil(data1.shape[0] / batch_size)))):
            dis = (A[batch_idx * batch_size: (batch_idx + 1) * batch_size] - B) ** 2.0
            dis = dis.sum(dim=-1).squeeze()
            choice_ = torch.argmin(dis, dim=1)
            choice_cluster[batch_idx * batch_size: (batch_idx + 1) * batch_size] = choice_
        choice_cluster = choice_cluster.long()
        return choice_cluster
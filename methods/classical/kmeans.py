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

from utils import config
from datasetLoader import ClusteringDataset

from .base import ClassicalMethod

class KMeans(ClassicalMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.max_iterations = cfg.get("KMeans", "max_iterations")

    def fit(self):
        if self.device.startswith("cuda"):
            rst = cudaKMeans(n_clusters=self.n_clusters, max_iter=self.max_iterations, seed=self.cfg["global"]["seed"]).fit_predict(torch.tensor(self.dataset.data, device=self.device).unsqueeze(0))
            # rst = rst.cpu().numpy().squeeze()
            return rst, self.dataset.data
        else:
            return skKMeans(n_clusters=self.n_clusters, max_iter=self.max_iterations, random_state=self.cfg["global"]["seed"]).fit_predict(self.dataset.data), self.dataset.data
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
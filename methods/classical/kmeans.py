import numpy as np
from logging import Logger
from utils import config
from sklearn.cluster import KMeans as skKMeans

from .base import ClassicalMethod

class KMeans(ClassicalMethod):
    def __init__(self, dataset, description, logger:Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.max_iterations = cfg.get("KMeans", "max_iterations")

    def fit(self):
        return skKMeans(n_clusters=self.k, max_iter=self.max_iterations).fit_predict(self.dataset.data), self.data
        # return kmeans(self.dataset.data, self.k, self.max_iterations)


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
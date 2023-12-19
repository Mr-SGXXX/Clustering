import numpy as np
from utils import config

class KMeans:
    def __init__(self, cfg: config):
        self.k = cfg.get("global", "n_clusters")
        self.max_iterations = cfg.get("KMeans", "max_iterations")

    def fit(self, data):
        return kmeans(data, self.k, self.max_iterations)

def kmeans(data, k, max_iterations=100):
    """
    K-means clustering algorithm.
    
    Args:
        data (array-like): Input data matrix of shape (n_samples, n_features).
        k (int): The number of clusters to form.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 100.
    """
    # Randomly select k initial cluster centers
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

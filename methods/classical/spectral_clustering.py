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
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as skSpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from logging import Logger
import typing
import numpy as np

from datasetLoader import ClusteringDataset
from utils import config

from .base import ClassicalMethod

class SpectralClustering(ClassicalMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.cut_type = cfg.get("SpectralClustering", "cut_type")
        self.distance_type = cfg.get("SpectralClustering", "distance_type")
        self.device = cfg.get("global", "device")
    
    def fit(self):
        # sc = skSpectralClustering(n_clusters=self.n_clusters, affinity=self.distance_type)
        # pred_labels = sc.fit_predict(self.dataset.data)
        # sc_embed = sc.affinity_matrix_.toarray()
        # return pred_labels, sc_embed
        return spectral_clustering(self.dataset.data, self.n_clusters, cut_type=self.cut_type, distance=self.distance_type, device=self.device)


def spectral_clustering(X, n_clusters, A=None, cut_type: typing.Literal["RatioCut", "NCut"] = 'NCut',
                        distance: typing.Union[typing.Callable, typing.Literal[
                            'nearest_neighbors',
                            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                            'braycurtis', 'canberra', 'chebyshev', 'correlation',
                            'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                            'minkowski', 'rogerstanimoto ', 'russellrao', 'seuclidean',
                            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']] = "euclidean",
                        device='cpu'):
    """
    Spectral clustering algorithm.

    Args:
        X (array-like): Input data matrix of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        A (array-like, optional): The similarity matrix of shape (n_samples, n_samples). If not provided, it will be computed based on the input data matrix X.
        cut_type (str, optional): The type of cut to use. Can be either "RatioCut" or "NCut". Defaults to "NCut".
        distance (str or callable, optional): The distance metric to use for computing the similarity matrix. Defaults to "euclidean".
        device (str, optional): The device to use for computation. Defaults to 'cpu'.

    Returns:
        array-like: Cluster labels for each sample.

    Raises:
        ValueError: If cut_type is not "RatioCut" or "NCut".

    """
    if A is None:
        if "nearest_neighbors" in distance:
             #Fit the NearestNeighbors model
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)

            # Get the K-nearest neighbors for each data point
            distances, indices = nbrs.kneighbors(X)

            # Initialize the adjacency matrix
            A = np.zeros((X.shape[0], X.shape[0]))

            # Set the entries in the adjacency matrix
            for i in range(X.shape[0]):
                A[i, indices[i]] = 1

            # Make the adjacency matrix symmetric
            A = 0.5 * (A + A.T)
        else:
            # Compute the similarity matrix
            A = pairwise_distances(X, metric=distance)
    else:
        assert A.shape == (X.shape[0], X.shape[0]), "A must be of shape (n_samples, n_samples)."
    A = torch.tensor(A, device=device)
    # Compute the graph Laplacian
    if cut_type == 'RatioCut':
        D = torch.diag(torch.sum(A, dim=1)).to(device)
        L = D - A
    elif cut_type == 'NCut':
        D = torch.diag(torch.sum(A, dim=1) ** (-0.5)).to(device)
        L = torch.eye(X.shape[0]).to(device) - torch.mm(torch.mm(D, A), D)
    else:
        raise ValueError(
            "Invalid cut_type. Must be either 'RatioCut' or 'NCut'.")

    # Compute the eigenvectors corresponding to the smallest eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # Sort the eigenvalues and eigenvectors in ascending order
    _, indices = torch.sort(eigenvalues, descending=False)
    sorted_eigenvectors = eigenvectors[:, indices]

    # Select the eigenvectors corresponding to the smallest eigenvalues
    selected_eigenvectors = sorted_eigenvectors[:, :n_clusters]

    # Normalize the eigenvectors
    normalized_eigenvectors = selected_eigenvectors
    # normalized_eigenvectors = F.normalize(selected_eigenvectors, p=2, dim=1)

    # Convert the normalized eigenvectors back to numpy array
    normalized_eigenvectors = normalized_eigenvectors.cpu().detach().numpy()

    # Cluster the normalized eigenvectors using K-means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(normalized_eigenvectors)

    # Return the cluster labels
    return kmeans.labels_, normalized_eigenvectors

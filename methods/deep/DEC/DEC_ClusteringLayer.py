# Copyright (c) 2023 Yuxuan Shao

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
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        """
        Clustering layer which computes the soft assignments of each sample to each cluster.

        :param n_clusters: number of clusters
        :param hidden_dim: hidden dimension, output of the encoder
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.mu = nn.Parameter(torch.zeros(self.n_clusters, self.hidden_dim, dtype=torch.float))

    def kmeans_init(self, data: torch.Tensor) -> None:
        """
        Initialize the cluster centers using k-means clustering.

        :param batch: batch of data
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        pred_label = kmeans.fit_predict(data.cpu().detach().numpy())
        mu = torch.from_numpy(kmeans.cluster_centers_)
        self.mu.data.copy_(mu)
        return pred_label

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment by student t-distribution.
        q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

        :param batch: batch of data
        :return: cluster assignment
        """
        q = 1.0 / (1.0 + (torch.sum((batch.unsqueeze(1) - self.mu).pow(2), dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        return (q.t() / torch.sum(q, dim=1)).t()
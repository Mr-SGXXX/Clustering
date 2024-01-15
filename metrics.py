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
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from logging import Logger
import torch
from tqdm import tqdm
from time import time


class Metrics:
    """
    Metrics for clustering performance evaluation.

    Args:
        with_ground_true (bool, optional): Whether to use ground truth for evaluation. Defaults to True.

    Methods:
        update_loss: Update loss.
        update: Update metrics.
        max: Get the max value of each metric.
        min: Get the min value of each metric.
        avg: Get the average value of each metric.
    """

    def __init__(self):
        self.ACC = AverageMeter("ACC")
        self.NMI = AverageMeter("NMI")
        self.ARI = AverageMeter("ARI")
        self.HOMO = AverageMeter("HOMO")
        self.COMP = AverageMeter("COMP")
        self.SC = AverageMeter("SC")
        self.PretrainLoss = {}
        self.Loss = {}
        self.pretrain_time_cost = 0
        self.clustering_time_cost = 0

    def update_pretrain_loss(self, **kwargs):
        """
        In each pretrain epoch, update every pretrain loss in the metrics, which will be used for generating pretrain loss figure.

        The loss name will be auto input in the type of the key of the parameter.
        """
        start_time = time()
        for key in kwargs.keys():
            if key not in self.PretrainLoss:
                self.PretrainLoss[key] = AverageMeter(key)
            self.PretrainLoss[key].update(kwargs[key])
        self.pretrain_time_cost += time() - start_time

    def update_loss(self, **kwargs):
        """
        In each clustering epoch, update every loss in the metrics, which will be used for generating loss figure.

        The loss name will be auto input in the type of the key of the parameter.
        """
        start_time = time()
        for key in kwargs.keys():
            if key not in self.Loss:
                self.Loss[key] = AverageMeter(key)
            self.Loss[key].update(kwargs[key])
        self.clustering_time_cost += time() - start_time

    def update(self, y_pred, features=None, y_true=None):
        """
        In each clustering epoch, update every metric in the metrics, which will be used for generating metrics figures.
        """
        start_time = time()
        assert features is None or type(features) is np.ndarray or type(features) is torch.Tensor
        if features is not None:
            # when the size of dataset is too big, calculate the sc costs too much time.
            sc = clusters_scores(y_pred, features)
            self.SC.update(sc)
        else:
            sc = None
        if y_true is not None:
            assert type(y_true) is np.ndarray, "y_true should be of type np.ndarray or None"
            acc, nmi, ari, homo, comp = evaluate(y_pred, y_true)
            self.ACC.update(acc)
            self.NMI.update(nmi)
            self.ARI.update(ari)
            self.HOMO.update(homo)
            self.COMP.update(comp)
        else:
            acc, nmi, ari, homo, comp = None, None, None, None, None
        self.clustering_time_cost += time() - start_time
        return (sc,), (acc, nmi, ari, homo, comp)

    def max(self):
        return (self.SC.max,), (self.ACC.max, self.NMI.max, self.ARI.max, self.HOMO.max, self.COMP.max)

    def min(self):
        return (self.SC.min,), (self.ACC.min, self.NMI.min, self.ARI.min, self.HOMO.min, self.COMP.min)

    def avg(self):
        return (self.SC.avg,), (self.ACC.avg, self.NMI.avg, self.ARI.avg, self.HOMO.avg, self.COMP.avg)

    def __str__(self):
        return  "Clustering Scores:\n" + \
                f"Last Epoch Scores: ACC: {self.ACC.last:.4f}\tNMI: {self.NMI.last:.4f}\tARI: {self.ARI.last:.4f}\n" + \
                f"Last Epoch Additional Scores: SC: {self.SC.last:.4f}\tHOMO: {self.HOMO.last:.4f}\tCOMP: {self.COMP.last:.4f}\n" + \
                f"Best Scores/Epoch: ACC: {self.ACC.max:.4f}/{self.ACC.argmax}\tNMI: {self.NMI.max:.4f}/{self.NMI.argmax}\tARI:{self.ARI.max:.4f}/{self.ARI.argmax}\n" + \
                f"Best Additional Scores/Epoch: SC: {self.SC.max:.4f}/{self.SC.argmax}\tHOMO: {self.HOMO.max:.4f}/{self.HOMO.argmax}\tCOMP: {self.COMP.max:.4f}/{self.COMP.argmax}"
        


class AverageMeter:
    def __init__(self, name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.val_list = []
        self.last = 0
        self.avg = 0
        self.max = float('-inf')
        self.argmax = 0
        self.min = float('inf')
        self.argmin = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, cnt=1):
        if val is not None:
            self.val_list.append(val)
            self.last = val
            self.sum += val * cnt
            self.cnt += cnt
            self.avg = self.sum / self.cnt
            if self.max < val:
                self.max = val
                self.argmax = self.cnt
            if self.min > val:
                self.min = val
                self.argmin = self.cnt

    def __str__(self) -> str:
        return f"{self.name}: Avg: {self.avg:.4f} Min: {self.min:.4f} Max: {self.max:.4f}"


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    # pdb.set_trace()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_pred, y_true):
    """
    Evaluate clustering performance. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    """
    assert y_pred is not None, "y_pred is necessary!"
    assert y_true is not None, "y_true is necessary!"
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, nmi, ari, homo, comp


def clusters_scores(y_pred, features):
    """
    Evaluate the intra-clusters and inter-clusters feature situation without ground truth
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        features: hidden features, numpy.ndarray with shape `(n_samples, hidden_dim)`
    """
    assert y_pred is not None, "y_pred is necessary!"
    assert features is not None, "When ground truth is not available, hidden feature is necessary!"
    assert type(features) is np.ndarray or type(
        features) is torch.Tensor, "features should be of type np.ndarray or torch.Tensor"
    if len(set(y_pred)) == 1:
        sc = -1.0
    else:
        sc = sklearn_silhouette_score(features.detach().cpu().numpy(), y_pred)
        # sc = silhouette_score(features.detach(), y_pred)
    return sc


def silhouette_score(X, labels, metric='euclidean'):
    """
    Compute the silhouette score in a vectorized way using PyTorch.

    Args:
    X (Tensor): Data points, size (n_samples, n_features)
    labels (Tensor): Cluster labels for each point, size (n_samples)
    metric (str): Metric used for distance computation, default is 'euclidean'

    Returns:
    float: Silhouette Score
    """
    # Convert labels to tensor if they are not
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)

    # Unique labels
    unique_labels = torch.unique(labels)

    # Initialize distances
    n = X.size(0)
    a = torch.zeros(n)
    b = torch.full((n,), float('inf'))

    # Calculate pairwise distance
    if metric == 'euclidean':
        dist_matrix = batched_cdist(X)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")

    # Compute intra-cluster distances (a)
    with tqdm(unique_labels, desc="Computing intra-cluster distances", dynamic_ncols=True, leave=False) as t:
        for label in t:
            mask = labels == label
            intra_cluster_distances = dist_matrix[mask][:, mask]
            sum_distances = torch.sum(intra_cluster_distances, dim=1)
            count = mask.sum() - 1  # Exclude the point itself
            a[mask] = sum_distances / count

    # Compute nearest cluster distance (b)
    with tqdm(unique_labels, desc="Computing inter-cluster distances", dynamic_ncols=True, leave=False) as t:
        for label in t:
            mask = labels == label
            for other_label in unique_labels:
                if label != other_label:
                    other_mask = labels == other_label
                    inter_cluster_distances = dist_matrix[mask][:, other_mask]
                    if inter_cluster_distances.nelement() != 0:
                        min_distances = torch.min(
                            inter_cluster_distances, dim=1)[0]
                        b[mask] = torch.min(b[mask], min_distances)

    # Handle clusters with single elements
    a[a == 0] = b[a == 0]

    # Silhouette values
    s = (b - a) / torch.max(a, b)

    # Average over all points
    return s.mean().item()


def batched_cdist(X, batch_size=2048):
    """
    Compute the pairwise distance matrix in batches to reduce memory usage.

    Args:
    X (Tensor): Data points, size (n_samples, n_features)
    batch_size (int): The size of each batch for computation

    Returns:
    Tensor: Pairwise distance matrix
    """
    n = X.size(0)
    dist_matrix = torch.zeros((n, n), device='cpu')

    with tqdm(range(0, n, batch_size), desc="Computing pairwise distance", dynamic_ncols=True, leave=False) as t:
        for i in t:
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                dist_matrix[i:end_i, j:end_j] = torch.cdist(
                    X[i:end_i], X[j:end_j]).to("cpu")
    return dist_matrix

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

import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

# This file is modified from https://github.com/facebookresearch/deepcluster

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def run_kmeans(x, nmb_clusters, logger=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if logger is not None:
        logger.info('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, logger=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, logger)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if logger is not None:
            logger.info('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, logger=None):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if logger is not None:
            logger.info('pic time: {0:.0f} s'.format(time.time() - end))
        return 0


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
from .kmeans import KMeans
from .spectral_clustering import SpectralClustering

"""
CLASSICAL_METHODS: A dict of classical clustering methods.

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, description, logger, cfg)`, `fit()` methods.
fit() should return the predicted labels and the features clustered.
the args must be the same as the base class in `base.py`.
"""

CLASSICAL_METHODS = {
    # A simple implement for K-means
    "KMeans": KMeans,
    # A simple implement for Spectral Clustering
    "SpectralClustering": SpectralClustering
}

"""
CLASSICAL_METHODS_INPUT_TYPES: A dict of classical clustering methods' input types.

The key is the name of the method, and the value is the flag list, and the flag means what the method can directly accept.
"""
CLASSICAL_METHODS_INPUT_TYPES = {
    "KMeans": ["seq"],
    "SpectralClustering": ["seq"]
}

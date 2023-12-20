from .kmeans import KMeans
from .spectral_clustering import SpectralClustering

"""
CLASSICAL_METHODS: A dict of classical clustering methods.

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(cfg)`, `fit(data)` methods.
fit() should return the predicted labels.
"""

CLASSICAL_METHODS = {
    # A simple implement for K-means
    "KMeans": KMeans,
    # A simple implement for Spectral Clustering
    "SpectralClustering": SpectralClustering
}

"""
CLASSICAL_METHODS_INPUT_IMG_FLAG: A dict of flags indicating whether the input data is an image.

The key is the name of the method, and the value is the flag.
"""
CLASSICAL_METHODS_INPUT_TYPES = {
    "KMeans": ["seq"],
    "SpectralClustering": ["seq"]
}

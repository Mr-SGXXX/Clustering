from kmeans import KMeans
from spectral_clustering import SpectralClustering

"""
CLASSICAL_METHODS: A dict of classical clustering methods.

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(cfg)`, `fit(data)` methods.
fit() should return the predicted labels.
"""

CLASSICAL_METHODS = {
    "KMeans": KMeans,
    "SpectralClustering": SpectralClustering
}

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

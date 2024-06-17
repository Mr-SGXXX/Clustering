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

# The following code refers to https://github.com/CSUBioGroup/scMAE/blob/main/datasets.py, thanks to the authors for their work.
import torch
import scanpy as sc
import numpy as np
import typing
import h5py
import scipy as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD


default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}
        
class IterativeSVDImputator(object):
    def __init__(self, svd_params=default_svd_params, iters=2):
        self.missing_values = 0.0
        self.svd_params = svd_params
        self.iters = iters
        self.svd_decomposers = [None for _ in range(self.iters)]

    def fit(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
            self.svd_decomposers[i].fit(transformed_X)
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]

    def transform(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]
        return transformed_X

def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
                  size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)  # 3
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / \
            np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(
            adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def load_scRNA_h5data(path, cfg):
    """Loads scRNA-seq dataset"""
    data_mat = h5py.File(
        f"{path}.h5", "r")
    X = np.array(data_mat['X'])
    Y = np.array(data_mat['Y'])
    return load_scRNA_data(X, Y, cfg)

    
def load_scRNA_data(X, Y, cfg):
    if Y.dtype != "int64":
        encoder_x = LabelEncoder()
        Y = encoder_x.fit_transform(Y)

    copy = cfg.get("XYh5_scRNA", "copy")
    nb_genes = cfg.get("XYh5_scRNA", "nb_genes")
    size_factors = cfg.get("XYh5_scRNA", "size_factors")
    normalize_input = cfg.get("XYh5_scRNA", "normalize_input")
    logtrans_input = cfg.get("XYh5_scRNA", "logtrans_input")

    X = np.ceil(X).astype(np.int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    adata = sc.AnnData(X)

    adata = normalize(adata,
                    copy=copy,
                    highly_genes=nb_genes,
                    size_factors=size_factors,
                    normalize_input=normalize_input,
                    logtrans_input=logtrans_input)
    sorted_genes = adata.var_names[np.argsort(adata.var["mean"])]
    adata = adata[:, sorted_genes]
    X = adata.X.astype(np.float32)

    if cfg.get("XYh5_scRNA", "SVD_impute"):
        imputator = IterativeSVDImputator(iters=2)
        imputator.fit(X)
        X = imputator.transform(X)

    return X, Y

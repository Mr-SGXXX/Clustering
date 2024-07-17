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

# This dataset loader is for loading scRNA data in h5 format with X and Y data, where X is the feature matrix and Y is the label matrix.
# By default, if the data is not found, it will attempt to download the data from the dataset_url dictionary.
# The dataset is then normalized and imputed using SVD imputation if the configuration is set to True.
# If the graph data is needed, the data can be loaded as a graph using the load_as_graph method.

# The online dataset is available at: https://zenodo.org/records/8175767 collected by the authors of the scMAE paper:
# Fang Z, Zheng R, Li M. scMAE: a masked autoencoder for single-cell RNA-seq clustering[J]. Bioinformatics, 2024, 40(1): btae020.
# Thanks to the authors for their work.

import torch
import numpy as np
import os
import h5py
import torch_geometric as pyg
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import dense_to_sparse
import typing
import scipy as sp
import scanpy as sc

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

from ..base import ClusteringDataset
from ..utils import download_dataset
from utils import config

    
default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}

dataset_url = {
    "Bach": "https://zenodo.org/records/8175767/files/Bach.h5?download=1",
    # Bach K, Pensa S, Grzelak M et al. Differentiation dynamics of mammaryepithelial cells revealed by single-cell RNA sequencing. Nat Commun 2017;8:2128.
    # Description: Mammary epithelial cells
    # Shape: (23184, 19965)
    # Label Number: 8
    # Platform: 10x
    
    "Baron_human": "https://zenodo.org/records/8175767/files/Baron.h5?download=1",
    # Baron M, Veres A, Wolock SL et al. A single-cell transcriptomic map of the human and mouse pancreas reveals inter-and intra-cell population structure. Cell Syst 2016;3:346–60.e4.
    # Description: Human pancreas
    # Shape: (8569, 20125)
    # Label Number: 14
    # Platform: inDrop
    
    "Guo": "https://zenodo.org/records/8175767/files/Guo.h5?download=1",
    # Guo J, Grow E J, Mlcochova H, et al. The adult human testis transcriptional cell atlas[J]. Cell research, 2018, 28(12): 1141-1157.
    # Description: Adult human testis
    # Shape: (6490, 27477)
    # Label Number: 12
    # Platform: 10x
    
    "hrvatin": "https://zenodo.org/records/8175767/files/hrvatin.h5?download=1",
    # Hrvatin S, Hochbaum DR, Nagy MA et al. Single-cell analysis of experience-dependent transcriptomic states in the mouse visual cortex. Nat Neurosci 2018;21:120–9.
    # Description: Mouse visual cortex
    # Shape: (48266, 25187)
    # Label Number: 8
    # Platform: Drop-seq
    
    "Limb_Muscle": "https://zenodo.org/records/8175767/files/Limb_Muscle.h5?download=1",
    # Tabula Muris Consortium, Overall C, Logistical C. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. Nature 2018; 562:367–72.
    # Description: Mouse limb muscle
    # Shape: (3909, 23341)
    # Label Number: 6
    # Platform: 10x
    
    "Quake_Smart-seq2_Lung": "https://zenodo.org/records/8175767/files/Quake_Smart-seq2_Lung.h5?download=1",
    # Tabula Muris Consortium, Overall C, Logistical C. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. Nature 2018; 562:367–72.
    # Description: Mouse lung
    # Shape: (1676, 23341)
    # Label Number: 11
    # Platform: Smart-seq2
    
    "Quake_10x_Spleen": "https://zenodo.org/records/8175767/files/Quake_10x_Spleen.h5?download=1",
    # Tabula Muris Consortium, Overall C, Logistical C. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. Nature 2018; 562:367–72.
    # Description: Mouse spleen
    # Shape: (9552, 23341)
    # Label Number: 5
    # Platform: 10x
    
    "Macosko": "https://zenodo.org/records/8175767/files/Macosko.h5?download=1",
    # Macosko EZ, Basu A, Satija R et al. Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets. Cell 2015;161:1202–14.
    # Description: Mouse retina cells
    # Shape: (44808, 23288)
    # Label Number: 12
    # Platform: Drop-seq
    
    "Melanoma_5K": "https://zenodo.org/records/8175767/files/Melanoma_5K.h5?download=1",
    # Tirosh, I., Izar, B., Prakadan, S. M., Wadsworth, M. H., Treacy, D., Trombetta, J. J., Rotem, A., Rodman, C., Lian, C., Murphy, G., et al. (2016). Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq. Science, 352(6282), 189–196.
    # Description: Metastatic melanoma
    # Shape: (4513, 23684)
    # Label Number: 9
    # Platform: smart-seq2
    
    "Pollen": "https://zenodo.org/records/8175767/files/Pollen.h5?download=1",
    # Pollen AA, Nowakowski TJ, Shuga J et al. Low-coverage single-cell mRNA sequencing reveals cellular heterogeneity and activated signaling pathways in developing cerebral cortex. Nat Biotechnol 2014;32:1053–8.
    # Description: Developing cerebral cortex
    # Shape: (301, 21721)
    # Label Number: 11
    # Platform: smart-seq2 (not for sure)
    
    "Shekhar": "https://zenodo.org/records/8175767/files/Shekhar.h5?download=1",
    # Shekhar K, Lapan S W, Whitney I E, et al. Comprehensive classification of retinal bipolar neurons by single-cell transcriptomics[J]. Cell, 2016, 166(5): 1308-1323. e30.
    # Description: Mouse retinal bipolar neurons
    # Shape: (26830, 13166)
    # Label Number: 18
    # Platform: Drop-seq
    
    "Tosches": "https://zenodo.org/records/8175767/files/Tosches.h5?download=1",
    # Tosches MA, Yamawaki TM, Naumann RK et al. Evolution of pallium, hippocampus, and cortical cell types revealed by single-cell transcriptomics in reptiles. Science 2018;360:881–8.
    # Description: Reptile brain
    # Shape: (18664, 23500)
    # Label Number: 15
    # Platform: Drop-seq
    
    "Wang": "https://zenodo.org/records/8175767/files/Wang.h5?download=1",
    # Wang Y, Tang Z, Huang H, et al. Pulmonary alveolar type I cell population consists of two distinct subtypes that differ in cell fate[J]. Proceedings of the National Academy of Sciences, 2018, 115(10): 2407-2412.
    # Description: Mouse pulmonart alveolar type I cells
    # Shape: (9519, 14561)
    # Label Number: 2
    # Platform: 10x
    
    "worm_neuron_cell": "https://zenodo.org/records/8175767/files/worm_neuron_cell.h5?download=1",
    # Cao J, Packer J S, Ramani V, et al. Comprehensive single-cell transcriptional profiling of a multicellular organism[J]. Science, 2017, 357(6352): 661-667.
    # Description: Worm neuron cells
    # Shape: (4186, 13488)
    # Label Number: 10
    # Platform: sci-RNA-seq
    
    "Young": "https://zenodo.org/records/8175767/files/Young.h5?download=1",
    # Young M D, Mitchell T J, Vieira Braga F A, et al. Single-cell transcriptomes from human kidneys reveal the cellular identity of renal tumors[J]. science, 2018, 361(6402): 594-599.
    # Description: Human kidney
    # Shape: (5685, 33658)
    # Label Number: 11
    # Platform: 10x (not for sure)
}

class XYh5_scRNA(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        super().__init__(cfg, needed_data_types)
        self.name = None

    def label_data_init(self):
        if 'seq' not in self.needed_data_types:
            raise ValueError(f"No available data type for XYh5_scRNA in {self.needed_data_types}")
        self.data_type = 'seq'
        data_name = self.cfg.get("XYh5_scRNA", "data_name")
        data_dir = os.path.join(self.cfg.get("global", "dataset_dir"), "XYh5_scRNA")
        data_path = os.path.join(data_dir, f"{data_name}.h5")
        if not os.path.exists(data_path):
            if data_name in dataset_url:
                download_dataset(dataset_url[data_name], data_path)
            else:
                raise FileNotFoundError(f"File {data_path} not found and no download URL provided for {data_name}")
        X, Y = load_scRNA_h5data(data_path, self.cfg)
        return X, Y
    
    def data_preprocess(self, sample):
        return sample
    
    def load_as_graph(self, weight_type:typing.Union[typing.Callable[[torch.Tensor], torch.Tensor], typing.Literal["cosine", "KNN", "Radius"]], **kwargs):
        return XYh5_scRNA_graph(self.cfg, self._XYh5_scRNA_graph_XY_loader, weight_type, transform=None, **kwargs)

    def _XYh5_scRNA_graph_XY_loader(self):
        return self.label_data, self.label


class XYh5_scRNA_graph(GraphDataset):
    def __init__(self, cfg, XY_loader, weight_type:typing.Union[typing.Callable[[torch.Tensor], torch.Tensor], typing.Literal["cosine", "KNN", "Radius"]], transform=None, **kwargs):
        root = cfg.get("global", "dataset_dir")
        self.data_name = cfg.get("XYh5_scRNA", "data_name")
        root = os.path.join(root, "XYh5_scRNA", self.data_name)
        self.weight_type = weight_type
        self.kwargs = kwargs
        if not os.path.exists(root):
            os.makedirs(root)
        super(XYh5_scRNA_graph, self).__init__(root, transform, pre_transform=XY_loader)
        self.data = torch.load(self.processed_paths[0])    

    def download(self):
        raise RuntimeError("Dataset not found")
    
    def process(self):
        if self.pre_transform is not None:
            X, Y = self.pre_transform()
            X, Y = torch.Tensor(X), torch.Tensor(Y)
        else:
            raise ValueError("Graph XY load failed")
        if self.transform is not None:
            X = self.transform(X)
        edge_index, edge_weight = self.adj_construct(X)
        data = GraphData(x=X, edge_index=edge_index, edge_weight=edge_weight, y=Y)
        torch.save(data, self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        weight_name = self.weight_type if type(self.weight_type) is str else self.weight_type.__name__
        return [f"{self.data_name}XY_{weight_name}_Graph.pt"]
    
    def adj_construct(self, X):
        if type(self.weight_type) == typing.Callable:
            adj_mat = self.weight_type(X, **self.kwargs)
        elif type(self.weight_type) is str:
            if self.weight_type == "cosine":
                X_norm = X / torch.norm(X, dim=1, keepdim=True)
                adj_mat = torch.mm(X_norm, X_norm.T)
                edge_index, edge_weight = dense_to_sparse(adj_mat)
            elif self.weight_type == "KNN":
                k = self.kwargs.get("k", None)
                assert k is not None, "KNN weight type requires k value to be set."
                edge_index = knn_graph(torch.Tensor(X), k, loop=True, **self.kwargs)
                edge_weight = None
            elif self.weight_type == "Radius":
                r = self.kwargs.get("r", None)
                assert r is not None, "Radius weight type requires r value to be set."
                edge_index = radius_graph(torch.Tensor(X), r, loop=True, **self.kwargs)
                edge_weight = None
            else:
                raise ValueError(f"Unknown weight type {self.weight_type}! Input a sparse matrix generation function or predefined weight type string!")
        else:
            raise ValueError(f"Unknown weight type {self.weight_type}! Input a sparse matrix generation function or predefined weight type string!")
                
        return edge_index, edge_weight
    
        
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
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

# This dataset loader is for loading scRNA data in h5 format
import torch
import numpy as np
import os
import h5py
import torch_geometric as pyg
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse
import typing

from ..base import ClusteringDataset
from .utils import load_scRNA_h5data
from utils import config

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
            raise FileNotFoundError(f"File {data_path} not found")
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
        root = os.path.join(root, "XYh5_scRNA", "graph", self.data_name)
        self.weight_type = weight_type
        self.kwargs = kwargs
        if not os.path.exists(root):
            os.makedirs(root)
        super(XYh5_scRNA_graph, self).__init__(root, transform, pre_transform=XY_loader)
        
        

    def download(self):
        raise NotImplementedError("Dataset not found")
    
    def process(self):
        if self.pre_transform is not None:
            self.X, self.Y = self.pre_transform()
            self.X, self.y = torch.Tensor(self.X), torch.Tensor(self.Y)
        else:
            raise ValueError("Graph XY load failed")
        self.edge_index, self.edge_weight = self.adj_construct("cosine")
        
    

    def get(self, idx):
        pass
    
    @property
    def processed_file_names(self):
        weight_name = self.weight_type if type(self.weight_type) is str else self.weight_type.__name__
        return [f"{self.data_name}XY_{weight_name}_Graph.pt"]

    
    
    def adj_construct(self):
        if type(self.weight_type) == typing.Callable:
            adj_mat = self.weight_type(self.X, **self.kwargs)
        elif type(self.weight_type) is str:
            if self.weight_type == "cosine":
                X_norm = self.X / torch.norm(self.X, dim=1, keepdim=True)
                adj_mat = torch.mm(X_norm, X_norm.T)
                edge_index, edge_weight = dense_to_sparse(adj_mat)
            elif self.weight_type == "KNN":
                k = self.kwargs.get("k", None)
                assert k is not None, "KNN weight type requires k value to be set."
                adj_mat = knn_graph(torch.from_numpy(self.X), k, loop=False)
                edge_index, edge_weight = dense_to_sparse(adj_mat)
            elif self.weight_type == "Radius":
                r = self.kwargs.get("r", None)
                assert r is not None, "Radius weight type requires r value to be set."
                adj_mat = radius_graph(torch.from_numpy(self.X), r, loop=False)
                edge_index, edge_weight = dense_to_sparse(adj_mat)
            else:
                raise ValueError(f"Unknown weight type {self.weight_type}! Input a sparse matrix generation function or predefined weight type string!")
        else:
            raise ValueError(f"Unknown weight type {self.weight_type}! Input a sparse matrix generation function or predefined weight type string!")
                
        return edge_index, edge_weight
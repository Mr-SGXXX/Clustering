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
import torch
import numpy as np
import os
import shutil
from torch_geometric.data.data import BaseData
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import download_google_url
from torch_geometric.utils import dense_to_sparse
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import typing
import zipfile

from datasetLoader.base import ClusteringDataset
from utils import config

from .utils import count_edges

class obgn_products(ClusteringDataset):
    def __init__(self, cfg:config, needed_data_types:list) -> None:
        super().__init__(cfg, needed_data_types)
        
    def label_data_init(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        self._graph = PygNodePropPredDataset(root=self.data_dir, name="ogbn-products")
        self._graph.edge_index = SparseTensor.from_edge_index(self._graph.edge_index)
        self._graph.y = self._graph.y[:, 0]
        self._graph.num_edges = count_edges(self._graph.edge_index)
        self._graph.edge_attr = None
        return self._graph.x.numpy(), self._graph.y.numpy()
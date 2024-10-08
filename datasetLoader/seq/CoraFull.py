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

# this dataset refers to "https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/tree/main/dataset" by Yue Liu
import torch
import numpy as np
import os
from torch_geometric.data.data import BaseData
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import download_google_url
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor
import typing
import zipfile

from ..base import ClusteringDataset
from utils import config

class CoraFull(ClusteringDataset):
    def __init__(self, cfg:config, needed_data_types:list) -> None:
        super().__init__(cfg, needed_data_types)
        
    def label_data_init(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        self._graph: GraphData = CoraFullGraph(root=self.data_dir).data
        return self._graph.x.numpy(), self._graph.y.numpy()
    

class CoraFullGraph(GraphDataset):
    google_id = "1XLqs084J3xgWW9jtbBXJOmmY84goT1CE"
    
    def __init__(self, root:str, transform: typing.Callable[..., typing.Any] = None) -> None:
        super().__init__(root, transform, pre_transform=None)
        self.data = torch.load(self.processed_paths[0])
        
    def download(self) -> None:
        if not os.path.exists(self.raw_paths[0]):
            download_google_url(self.google_id, self.raw_dir, "corafull.zip")
        with zipfile.ZipFile(self.raw_paths[0], 'r') as f:
            f.extractall(self.raw_dir)
        
    @property
    def raw_file_names(self) -> str:
        return ["corafull.zip", "corafull_adj.npy", "corafull_feat.npy", "corafull_label.npy"]
    
    @property
    def processed_file_names(self) -> str:
        return "corafull.pt"
    
    def process(self) -> None:
        X = torch.tensor(np.load(os.path.join(self.raw_dir, "corafull_feat.npy")), dtype=torch.float)
        Y = torch.tensor(np.load(os.path.join(self.raw_dir, "corafull_label.npy")), dtype=torch.long)
        adj = torch.tensor(np.load(os.path.join(self.raw_dir, "corafull_adj.npy")), dtype=torch.float)
        
        adj_t = SparseTensor.from_dense(adj)
        
        data = GraphData(x=X, y=Y, edge_index=adj_t)
        
        torch.save(data, self.processed_paths[0])
        
    def get(self, idx: int) -> BaseData:
        if idx != 0:
            raise IndexError
        return self.data
    
    def len(self) -> int:
        return 1

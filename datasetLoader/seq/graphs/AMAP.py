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
import shutil
from torch_geometric.data.data import BaseData
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import download_google_url
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor
import typing
import zipfile

from datasetLoader.base import ClusteringDataset
from utils import config

class AMAP(ClusteringDataset):
    def __init__(self, cfg:config, needed_data_types:list) -> None:
        super().__init__(cfg, needed_data_types)
        
    def label_data_init(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        self._graph: GraphData = AMAPGraph(root=self.data_dir).data
        return self._graph.x.numpy(), self._graph.y.numpy()
    

class AMAPGraph(GraphDataset):
    google_id = "1qqLWPnBOPkFktHfGMrY9nu8hioyVZV31"
    
    def __init__(self, root:str, transform: typing.Callable[..., typing.Any] = None) -> None:
        super().__init__(root, transform, pre_transform=None)
        self.data = torch.load(self.processed_paths[0])
        
    def download(self) -> None:
        if not os.path.exists(self.raw_paths[0]):
            download_google_url(self.google_id, self.raw_dir, "amap.zip")
        with zipfile.ZipFile(self.raw_paths[0], 'r') as f:
            f.extractall(self.raw_dir)
        
    @property
    def raw_file_names(self) -> str:
        return ["amap.zip", "amap_adj.npy", "amap_feat.npy", "amap_label.npy"]
    
    @property
    def processed_file_names(self) -> str:
        return "amap.pt"
    
    def process(self) -> None:
        X = torch.tensor(np.load(os.path.join(self.raw_dir, "amap_feat.npy")), dtype=torch.float)
        Y = torch.tensor(np.load(os.path.join(self.raw_dir, "amap_label.npy")), dtype=torch.long)
        adj = torch.tensor(np.load(os.path.join(self.raw_dir, "amap_adj.npy")), dtype=torch.float)
        
        adj_t = SparseTensor.from_dense(adj)
        
        data = GraphData(x=X, y=Y, edge_index=adj_t)
        
        torch.save(data, self.processed_paths[0])
        
    def get(self, idx: int) -> BaseData:
        if idx != 0:
            raise IndexError
        return self.data
    
    def len(self) -> int:
        return 1
    
    
    
if __name__ == "__main__":
    # print(np.load("data/raw/wiki_feat.npy").astype(np.float32))
    data = AMAPGraph(root="data").data
    # 输出各个属性的 shape
    print(f'Number of nodes: {data.x.shape[0]}')  # 节点数量
    print(f'Number of features per node: {data.x.shape[1]}')  # 每个节点的特征数量
    print(f'Number of edges: {data.edge_index.nnz() // 2}')  # 边的数量
    print(f'Number of node labels: {data.y.shape[0]}')  # 节点标签数量
    print(f'Number of classes: {data.y.unique().shape[0]}')  # 类别数量

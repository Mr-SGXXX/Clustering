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
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data.data import BaseData
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
import typing
import os

from utils import config


class ClusteringDataset(Dataset):
    """
    Base class for all datasets.
    The detailed dataset should be set and procesed based on this class.
    When you need to implement a new dataset, you need to inherit this class and implement the `data_init` and `data_preprocess` methods.
    When you use unlabeled data, you also need to implement the `unlabeled_data_init` method.

    When you want to change the label, just set the `label` attribute to the new label.

    Args:
        cfg (config): Config class.
        needed_data_types (list): A list of data types that are needed.

    Attributes:
        cfg (config): Config class implemented in `utils/config.py`.
        name (str): The name of the dataset, default to the class name.
        needed_data_types (list): A list of data types that the method need.
        label_data (torch.Tensor or np.ndarray): Labeled data. Need to be set in `label_data_init`.
        label (torch.Tensor or np.ndarray): Labels of the labeled data. Need to be set in `label_data_init`.
        unlabel_data (torch.Tensor or np.ndarray): Unlabeled data. Need to be set in `unlabeled_data_init`.
        label_length (int): Length of the labeled data. Automatically calculated when used.
        unlabel_length (int): Length of the unlabeled data. Automatically calculated when used.
        num_classes (int): Number of classes in the dataset. Automatically calculated when used.
        
    Methods:
        label_data_init() -> Union[torch.Tensor, np.ndarray, list of path], Union[torch.Tensor, np.ndarray, list of label]: 
            Used to initialize the `label_data` and `label` of the dataset.
            Necessary to be implemented if you want to use the dataset with label.
        unlabeled_data_init -> Union[torch.Tensor, np.ndarray, list of path]: 
            Used to initialize the `unlabel_data` of the dataset.
            Necessary to be implemented if you want to use the dataset with unlabeled data.
        data_preprocess -> Union[torch.Tensor, np.ndarray]: 
            Preprocess the every data loaded in `data_init` and `unlabeled_data_init`, return the preprocessed data, `torch.Tensor` or `np.ndarray`. Default to return the sample directly.
        use_full_data -> None:
            Use the whole dataset, usually for pretraining.
        use_label_data -> None:
            Use the labeled data, usually for clustering.
        load_graph_XY -> Tuple[torch.Tensor, torch.Tensor]:
            Generate the X and Y data of the graph, return the X and Y data, (torch.Tensor, torch.Tensor).
            Need to be rewritten if you want to load the data as a graph but not use the default data and label.
        to_graph -> torch_geometric.data.Dataset:
            Load the data as a graph (KNN default), return the graph data, `torch_geometric.data.Dataset`.

    """

    def __init__(self, cfg: config, needed_data_types: list):
        """
        Initialize the dataset.
        """
        self.cfg = cfg
        self.name = self.__class__.__name__
        self.needed_data_types = needed_data_types
        self.data_dir = os.path.join(self.cfg.get("global", "dataset_dir"), self.name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self._use_full_data = True
        self._label_data = None
        self._label = None
        self._unlabel_data = None
        self._label_len = None
        self._unlabel_len = None
        self._num_classes = None
        self._input_dim = None
        self._graph = None
        self._data_changed = False
        assert self.label_data is not None or self.unlabel_data is not None, "No available data"

    def __len__(self) -> int:
        """
        return the length of the dataset, int

        In the pretrain mode, if there is any unlabeled data, return the length of all data
        In the clustering mode, return the length of the labeled data
        """
        if self._use_full_data or self.label is None:
            return self.label_length + self.unlabel_length
        else:
            return self.label_length
        
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, typing.Union[torch.Tensor, None], torch.Tensor]:
        """
        return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor), 
        
        those unlabeled data should be the last to return

        for those unlabeled data, return (torch.Tensor, None, torch.Tensor)
        """
        data = self._data_preprocess(self.label_data[index] if index < self.label_length else self.unlabel_data[index - self.label_length])
        if type(data) is not torch.Tensor and type(data) is not np.ndarray:
            raise ValueError(
                "The data should be torch.Tensor or np.ndarray after the data preprocess")
        if index < self.label_length:
            return torch.from_numpy(np.array(data)), \
                torch.from_numpy(np.array(self.label[index])), \
                torch.from_numpy(np.array(index))
        else:
            return torch.from_numpy(np.array(data)), None, torch.from_numpy(np.array(index))

    def data_preprocess(self, sample) -> typing.Union[torch.Tensor, np.ndarray]:
        """
        preprocess a single sample, return the preprocessed data, torch.Tensor or np.ndarray

        When you need to preprocess the data of single sample (the data is not a Tensor when loaded in `data_init`), you need to overwrite this method
        """
        return sample

    def _data_preprocess(self, sample) -> typing.Union[torch.Tensor, np.ndarray]:
        """
        preprocess a single sample, return the preprocessed data, torch.Tensor or np.ndarray

        When you need to preprocess the data of single sample (the data is not a Tensor when loaded in `data_init`), you need to overwrite `data_preprocess` method
        """
        sample = self.data_preprocess(sample)
        assert type(sample) is torch.Tensor or type(sample) is np.ndarray, "The sample should be torch.Tensor or np.ndarray, consider to overwrite the data_preprocess method to preprocess the data"
        return torch.tensor(sample)

    def use_full_data(self):
        """
        use the full dataset (both labeled and unlabeled data), usually for pretraining,
        
        not suggested to change the default implementation
        """
        self._use_full_data = True

    def use_label_data(self):
        """
        use the labeled data, usually for clustering,
        
        not suggested to change the default implementation
        """
        if self.label is None:
            print("There is no available label data, use the full data instead")
            self._use_full_data = True
        else:
            self._use_full_data = False


    def label_data_init(self) -> typing.Tuple[typing.Union[torch.Tensor, np.ndarray, list, None], typing.Union[torch.Tensor, np.ndarray, None]]:
        """
        initialize the `data` of the dataset as `np.ndarray` or `torch.Tensor`, and the `label` of the dataset as `np.ndarray` or `torch.Tensor`.

        if the dataset loads each data in the __getitem__ method, the `data_init` is suggested to be implemented
        """
        return None, None

    def unlabeled_data_init(self) -> typing.Union[torch.Tensor, np.ndarray, list, None]:
        """
        initialize the `unlabel_data` of the dataset as `np.ndarray`, it should return the `np.ndarray`.

        if the dataset loads each data in the __getitem__ method, the `unlabeled_data_init` is suggested to be implemented
        """
        return None

    def load_graph_XY(self) -> typing.Tuple[typing.Union[torch.Tensor, np.ndarray], typing.Union[torch.Tensor, np.ndarray, None]]:
        """
        an optional method to generate the X and Y data of the graph,
        return the X and Y data, (torch.Tensor, torch.Tensor)
        
        Need to be rewritten if you want to load the data as a graph but not use the default data and label.
        """
        if self.label_data is not None and self.unlabel_data is not None and self._use_full_data:
            return self.data, torch.cat(self.label, -torch.ones(self.unlabel_length, dtype=torch.long))
        elif self.label_data is not None:
            return self.data, self.label
        else:
            return self.data, None
        # raise NotImplementedError("The load_graph_XY method should be implemented if you want to load the data as a graph")

    def to_graph(self, data_dir=None, data_desc=None, 
                 graph_type:typing.Union[typing.Callable[..., torch.Tensor], typing.Literal["KNN", "Radius"]]="KNN", 
                 distance_type:typing.Union[typing.Literal["Euclidean", "Cosine", "Heat", "NormCos", "Manhattan"]]="Euclidean",
                 **kwargs) -> GraphData:
        """
        method to load the data as a graph, return the graph data, torch_geometric.data.Dataset
        if the graph data is initialized by using "_graph", return the initialized graph data directly
        Args:
            data_dir (str): the directory of the graph data to save
            data_desc (str, Optional): the additional desp of the data, default to None
            graph_type (Union[Callable[[torch.Tensor], torch.Tensor], Literal["KNN", "Radius"]]): 
                the weight type of the graph, default to "KNN", with default k=10. 
                You can also input a function to generate the adjacency matrix.
            distance_type (Union[Literal["Euclidean", "Cosine", "Heat", "NormCos", "Manhattan"]]): 
                the distance type of the graph, default to "Euclidean"
            **kwargs: the parameters of the weight type function
        """
        if data_dir is None:
            data_dir = self.data_dir
        if self._graph is None:
            assert not (self._data_changed and data_desc is None), "When you change the original data, you need to give a data description"
            self._graph = Graph(self.load_graph_XY, data_dir, data_desc, graph_type, distance_type, transform=None,**kwargs).data
        elif self._data_changed:
            self._graph.x = self.data
        return self._graph

    @property
    def label_length(self):
        """
        the label length of the dataset, int, automatically calculated when used
        """
        if self._label_len is None:
            if self.label_data is None:
                self._label_len = 0
            raise ValueError(
                "Directly use the `label_length` or set `data` first")
        return self._label_len

    @property
    def unlabel_length(self):
        """
        the length of the unlabel data, int, default to 0, automatically calculated when used
        """
        if self._unlabel_len is None:
            if self.unlabel_data is None:
                self._unlabel_len = 0
        return self._unlabel_len

    @property
    def label_data(self) -> typing.Union[torch.Tensor, np.ndarray, None]:
        """
        all the labeled data of the dataset, numpy.ndarray

        Automatically calculated when used by `label_data_init`
        """
        if self._label_data is None:
            self._label_data, self._label = self.label_data_init()
            if isinstance(self._label, torch.Tensor):
                self._label = self._label.to(dtype=torch.long)
                self._label = self._label.cpu().detach().numpy()
            elif isinstance(self._label, np.ndarray):
                self._label = self._label.astype(np.int64)
            if self._label_data is not None:
                self._label_len = len(self._label_data)
                if self._input_dim is None:
                    if type(self._label_data) is np.ndarray or type(self._label_data) is torch.Tensor:
                        self._input_dim = list(self._label_data.shape[1:])
                    else:
                        self._input_dim = list(self._data_preprocess(self._label_data[0]).shape)
                    if len(self._input_dim) == 1:
                        self._input_dim = self._input_dim[0]
                else:
                    assert self._input_dim == self._label_data.shape[1:] if type(self._input_dim) is list else self._input_dim == self._label_data.shape[1], "The shape of the data should be the same as the input dim"
        if isinstance(self._label_data, torch.Tensor) or isinstance(self._label_data, np.ndarray):
            return torch.tensor(self._label_data)
        elif self._label_data is None:
            return None
        else:
            return torch.tensor(np.array([self._data_preprocess(raw_data) for raw_data in self._label_data]))

    @property
    def label(self) -> typing.Union[torch.Tensor, np.ndarray, None]:
        """
        all the labels of the labeled data, numpy.ndarray

        Automatically calculated when used by `label_data_init`
        """
        if self._label is None:
            if self.label_data is not None:
                self._label_len = len(self._label)
        return self._label
    
    @label.setter
    def label(self, new_label):
        assert len(new_label) == len(self._label_data), "The length of the label should be the same as the data"
        self._num_classes = None
        if isinstance(new_label, torch.Tensor):
            new_label = new_label.to(dtype=torch.long)
            new_label = new_label.cpu().detach().numpy()
        elif isinstance(new_label, np.ndarray):
            new_label = new_label.astype(np.int64)
        self._label = new_label

    @property
    def unlabel_data(self) -> typing.Union[torch.Tensor, np.ndarray, None]:
        """
        the unlabel data of the dataset, if no unlabel data, set it to None, numpy.ndarray, default to None
        
        commonly, the unlabel data is only used in pretraining.
        
        remember that the unlabel data must be the last part in the self.data
        
        if the dataset loads each data in the __getitem__ method, the `unlabeled_data_init` is suggested to be implemented
        """
        if self._unlabel_data is None:
            self._unlabel_data = self.unlabeled_data_init()
            if self._unlabel_data is not None:
                self._unlabel_len = len(self._unlabel_data)
                if self._input_dim is None:
                    if type(self._unlabel_data) is np.ndarray or type(self._unlabel_data) is torch.Tensor:
                        self._input_dim = list(self._unlabel_data.shape[1:])
                    else:
                        self._input_dim = list(self.data_preprocess(self._unlabel_data[0]).shape)
                    if len(self._input_dim) == 1:
                        self._input_dim = self._input_dim[0]
                else:
                    assert self._input_dim == self._unlabel_data.shape[1:] if type(self._input_dim) is list else self._input_dim == self._unlabel_data.shape[1], "The shape of the data should be the same as the input dim"
        if isinstance(self._unlabel_data, torch.Tensor) or isinstance(self._unlabel_data, np.ndarray):
            return torch.tensor(self._unlabel_data)
        elif self._unlabel_data is None:
            return None
        else:
            return torch.tensor(np.array([self.data_preprocess(raw_data) for raw_data in self._unlabel_data]))
    
    @property
    def num_classes(self) -> int:
        """
        the number of classes in the dataset, int
        """
        if self._num_classes is None and self.label is not None:
            self._num_classes = len(np.unique(self.label))
        return self._num_classes
    
    @property
    def input_dim(self) -> typing.Tuple[int]:
        """
        the input dimension of the dataset, tuple of int
        """
        if self._input_dim is None:
            if self.label_data is None and self.unlabel_data is None:
                raise ValueError("No available data")
        return self._input_dim
    
    @property
    def data(self) -> torch.Tensor:
        """
        the processed data of the whole dataset, (torch.Tensor, torch.Tensor)
        """
        if self.label_data is not None:
            if isinstance(self.label_data, torch.Tensor):
                rst = self.label_data
            else:
                rst = torch.stack([self._data_preprocess(raw_data) for raw_data in self.label_data], dim=0)
            if self.unlabel_data is not None and self._use_full_data:
                if isinstance(self.unlabel_data, torch.Tensor):
                    rst = torch.cat([rst, self.unlabel_data], dim=0)
                else:
                    rst = torch.cat([rst, torch.stack([self._data_preprocess(raw_data) for raw_data in self.unlabel_data], dim=0)], dim=0)
            return rst
        else:
            return torch.stack([self._data_preprocess(raw_data) for raw_data in self.unlabel_data], dim=0)

    
    @data.setter
    def data(self, new_data:torch.Tensor):
        assert len(new_data) == self.label_length + (self.unlabel_length if self.unlabel_data is not None and self._use_full_data else 0), "The length of the new data should be the same as the original data"
        self._data_changed = True
        self._input_dim = new_data.shape[1:]
        self._label_data = new_data[:self.label_length]
        if self.unlabel_data is not None and self._use_full_data:
            self._unlabel_data = new_data[self.label_length:]

    
class Graph(GraphDataset):
    """
    Used to convert the default dataset to a graph dataset.

    Args:
        data_dir (str): the directory of the graph data to save
        data_desp (str, optional): Additional description for the data. Defaults to None.
        XY_loader (Callable): the function to load the data, return the X and Y data
        graph_type (Union[Callable[[torch.Tensor], torch.Tensor], Literal["cosine", "KNN", "Radius"]]): 
            the weight type of the graph, default to "KNN". You can also input a function to generate the weight matrix.
        transform (Callable): the transform function of the data before calculating graph adjacency matrix, default to None
        **kwargs: the parameters of the weight type function
        
    Attributes:

    """
    def __init__(self, XY_loader, data_dir, data_desc=None, 
                 graph_type:typing.Union[typing.Callable[..., torch.Tensor], typing.Literal["KNN", "Radius"]]= "KNN",
                 distance_type:typing.Union[typing.Literal["Euclidean", "Cosine", "Heat", "NormCos"]]="Euclidean", 
                 transform:typing.Callable=None, **kwargs):
        self.data_name = os.path.basename(data_dir).split()[0]
        if data_desc is not None:
            self.data_name += "_" + data_desc
        self.graph_type = graph_type
        self.distance_type = distance_type
        self.kwargs = kwargs
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        super(Graph, self).__init__(data_dir, transform, pre_transform=XY_loader)
        self.data:GraphData = torch.load(self.processed_paths[0]) # contains only one graph data constructed by the whole dataset   

    def download(self):
        raise RuntimeError("Dataset not found. Please put the data file in the correct directory first.")
    
    def process(self):
        if self.pre_transform is not None:
            X, Y = self.pre_transform()
            X = torch.Tensor(X)
            if Y is not None:
                Y = torch.Tensor(Y)
        else:
            raise ValueError("Graph XY load failed.")
        if self.transform is not None:
            X = self.transform(X)
        edge_index, edge_weight = self.adj_construct(X)
        adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(X.size(0), X.size(0)))
        data = GraphData(x=X, edge_index=adj_t, y=Y)
        torch.save(data, self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        weight_name = self.graph_type if type(self.graph_type) is str else self.graph_type.__name__
        v = self.kwargs.get("k", 5) if self.graph_type == "KNN" else self.kwargs.get("r", None)
        assert v is not None, "Radius graph requires r value to be set."
        return [f"{self.data_name}_{weight_name}_{v}_Graph.pt"]
    
    @property
    def raw_file_names(self):
        return []
    
    def download(self):
        pass
    
    def adj_construct(self, X):
        if isinstance(self.graph_type, typing.Callable):
            adj_mat = self.graph_type(X, **self.kwargs)
            assert isinstance(adj_mat, torch.Tensor) and adj_mat.size(0) == adj_mat.size(1) == X.size(0) or type(adj_mat) is typing.Iterable and len(adj_mat) == 2, \
                "The weight matrix should be a torch.Tensor with the shape of (num_samples, num_samples) or a tuple of (edge_index, edge_weight)"
            if type(adj_mat) is torch.Tensor:
                assert adj_mat.size(0) == adj_mat.size(1), "The weight matrix should be a square matrix"
                edge_index, edge_weight = dense_to_sparse(adj_mat)
            else:
                edge_index, edge_weight = adj_mat
                assert type(edge_index) is torch.Tensor and (edge_weight is None or type(edge_weight) is torch.Tensor), "The edge index should be torch.Tensor and edge weight should be None or torch.Tensor"
                if edge_weight is not None:
                    assert edge_index.size(1) == edge_weight.size(0), "The edge index and edge weight should have the same length"
        elif isinstance(self.graph_type, str):
            assert len(X.shape) == 2, "The input data should be sequence, with shape (num_samples, num_features)"
            if self.graph_type == "KNN":
                k = self.kwargs.get("k", 5)
                if self.distance_type == "Euclidean":
                    edge_index = knn_graph(torch.Tensor(X), k, **self.kwargs)
                    edge_weight = None
                elif self.distance_type == "Cosine":
                    edge_index = knn_graph(torch.Tensor(X), k, cosine=True, **self.kwargs)
                    edge_weight = None
                else:
                    dist = self.cal_distance(X)
                    k = self.kwargs.get("k", 5)
                    
                    inds = []
                    for i in range(dist.shape[0]):
                        ind = np.argpartition(dist[i, :], k+1)[:k+1]
                        inds.append(ind)
                    
                    adj_mat = np.zeros_like(dist)
                    
                    for i, v in enumerate(inds):
                        for vv in v:
                            if vv == i:
                                pass
                            else:
                                adj_mat[i, vv] = 1
                                adj_mat[vv, i] = 1
                            
                    adj_mat = torch.tensor(adj_mat)
                    if self.kwargs.get("loop", False):
                        adj_mat.fill_diagonal_(1) # add self-loop
                    edge_index, _ = dense_to_sparse(adj_mat)
                    edge_weight = None
            elif self.graph_type == "Radius":
                r = self.kwargs.get("r", None)
                assert r is not None, "Radius graph requires r value to be set."
                if self.distance_type == "Euclidean":
                    edge_index = radius_graph(torch.Tensor(X), r, **self.kwargs)
                    edge_weight = None
                else:
                    dist = self.cal_distance(X)
                    adj_mat = dist < r
                    adj_mat = torch.tensor(adj_mat)
                    if not self.kwargs.get("loop", False):
                        adj_mat.fill_diagonal_(0) # remove self-loop
                    edge_index, _ = dense_to_sparse(adj_mat)
                    edge_weight = None
            else:
                raise ValueError(f"Unknown weight type {self.graph_type}! Input a sparse matrix generation function or predefined weight type string!")
        else:
            raise ValueError(f"Unknown weight type {self.graph_type}! Input a sparse matrix generation function or predefined weight type string!")
                
        return edge_index, edge_weight
    
    def cal_distance(self, X):
        # if self.distance_type == "Cosine":
        #     X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        #     dist = 1 - np.dot(X_norm, X_norm.T)
        if self.distance_type == "NormCos":
            X_binary = np.where(X > 0, 1, 0)
            X_normalized = normalize(X_binary, axis=1, norm="l1")
            dist = 1 - np.dot(X_normalized, X_normalized.T)
        elif self.distance_type == "Heat":
            dist = -0.5 * pair(X) ** 2
            dist = 1 - np.exp(dist)
        elif self.distance_type == "Manhattan":
            dist = pair(X, metric="manhattan")
        else:
            raise ValueError(f"Unknown distance type {self.distance_type}! Input a predefined distance type string!")
        return dist
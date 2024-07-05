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
import torch
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Dataset as PygDataset
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
        cfg (config): Config class.
        needed_data_types (list): A list of data types that the method need.
        label_data (torch.Tensor or np.ndarray): Labeled data. Need to be set in `label_data_init`.
        label (torch.Tensor or np.ndarray): Labels of the labeled data. Need to be set in `label_data_init`.
        unlabel_data (torch.Tensor or np.ndarray): Unlabeled data. Need to be set in `unlabeled_data_init`.
        total_length (int): Length of all data. Automatically calculated when used.
        unlabel_length (int): Length of the unlabeled data. Automatically calculated when used.
        num_classes (int): Number of classes in the dataset. Automatically calculated when used.

    Methods:
        label_data_init() -> Union[torch.Tensor, np.ndarray, list of path], Union[torch.Tensor, np.ndarray, list of label]: 
            Used to initialize the `label_data` and `label` of the dataset.
        unlabeled_data_init -> Union[torch.Tensor, np.ndarray, list of path]: 
            Used to initialize the `unlabel_data` of the dataset.
        data_preprocess -> Union[torch.Tensor, np.ndarray]: 
            Preprocess the every data loaded in `data_init` and `unlabeled_data_init`, return the preprocessed data, `torch.Tensor` or `np.ndarray`. Default to return the sample directly.
        
    """

    def __init__(self, cfg: config, needed_data_types: list):
        """
        Initialize the dataset.
        """
        self.cfg = cfg
        self.needed_data_types = needed_data_types
        self._full_data = True
        self._label_data = None
        self._label = None
        self._unlabel_data = None
        self._label_len = None
        self._unlabel_len = None
        self._num_classes = None
        self._input_dim = None

    def __len__(self) -> int:
        """
        return the length of the dataset, int

        In the pretrain mode, if there is any unlabeled data, return the length of all data
        In the clustering mode, return the length of the labeled data
        """
        if self._full_data or self.label is None:
            return self.label_length + self.unlabel_length
        else:
            return self.label_length
        
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, typing.Union[torch.Tensor, None], torch.Tensor]:
        """
        return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor), 
        
        those unlabeled data should be the last to return

        for those unlabeled data, return (torch.Tensor, None, torch.Tensor)
        """
        data = self.data_preprocess(self.label_data[index])
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

        When you need to preprocess the data of single sample (the data is not a Tensor when loaded in `data_init`), you need to override this method
        """
        return sample

    def use_full_data(self):
        """
        set the dataset to pretrain mode, not suggested to change the default implementation
        """
        self._full_data = True

    def use_label_data(self):
        """
        set the dataset to clustering mode, not suggested to change the default implementation
        """
        if self.label is None:
            print("There is no available label data, use the full data instead")
            self._full_data = True
        else:
            self._full_data = False


    def label_data_init(self) -> typing.Tuple[typing.Union[torch.Tensor, np.ndarray], typing.Union[torch.Tensor, np.ndarray]]:
        """
        initialize the `data` of the dataset as `np.ndarray` or `torch.Tensor`, and the `label` of the dataset as `np.ndarray` or `torch.Tensor`.

        if the dataset loads each data in the __getitem__ method, the `data_init` is suggested to be implemented
        """
        return None, None

    def unlabeled_data_init(self):
        """
        initialize the `unlabel_data` of the dataset as `np.ndarray`, it should return the `np.ndarray`.

        if the dataset loads each data in the __getitem__ method, the `unlabeled_data_init` is suggested to be implemented
        """
        return None
    
    def load_as_graph(self) -> PygDataset:
        """
        An optional method to load the data as a graph, return the graph data, torch_geometric.data.Data
        """
        raise NotImplementedError("The method should be implemented in the dataset class")

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
            else:
                raise ValueError(
                    "Directly use the `unlabel_length` or set `unlabel_data` first")
        return self._unlabel_len

    @property
    def label_data(self) -> typing.Union[torch.Tensor, np.ndarray, None]:
        """
        all the labeled data of the dataset, numpy.ndarray

        Automatically calculated when used by `label_data_init`
        """
        if self._label_data is None:
            self._label_data, self._label = self.label_data_init()
            if self._label_data is not None:
                self._label_len = len(self._label_data)
                if self._input_dim is None:
                    if type(self._label_data) is np.ndarray or type(self._label_data) is torch.Tensor:
                        self._input_dim = self._label_data.shape[1:]
                    else:
                        self._input_dim = self.data_preprocess(self._label_data[0]).shape
                    if len(self._input_dim) == 1:
                        self._input_dim = self._input_dim[0]
                else:
                    assert self._input_dim == self._label_data.shape[1:] if type(self._input_dim) is list else self._input_dim == self._label_data.shape[1], "The shape of the data should be the same as the input dim"
        return self._label_data

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
                        self._input_dim = self._unlabel_data.shape[1:]
                    else:
                        self._input_dim = self._unlabel_data.shape[1:]
                    if len(self._input_dim) == 1:
                        self._input_dim = self._input_dim[0]
                else:
                    assert self._input_dim == self._unlabel_data.shape[1:] if type(self._input_dim) is list else self._input_dim == self._unlabel_data.shape[1], "The shape of the data should be the same as the input dim"
        return self._unlabel_data
    
    @property
    def num_classes(self) -> int:
        """
        the number of classes in the dataset, int
        """
        if self._num_classes is None:
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

    

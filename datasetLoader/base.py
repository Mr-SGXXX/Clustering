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
import numpy as np
import typing

from utils import config


class ClusteringDataset(Dataset):
    """
    Base class for all datasets.

    Args:
        cfg (config): Config class.
        needed_data_types (list): A list of data types that are needed.

    Attributes:
        cfg (config): Config class.
        unlabel_data (torch.Tensor or np.ndarray): Unlabeled data.
        data (torch.Tensor or np.ndarray): Labeled data.
        total_length (int): Length of all data.
        unlabel_length (int): Length of the unlabeled data.
        label (torch.Tensor or np.ndarray): Labels of the labeled data.

    Methods:
        pretrain: Set the dataset to pretrain mode.
        clustering: Set the dataset to clustering mode.
    """

    def __init__(self, cfg: config, needed_data_types: list):
        """
        Initialize the dataset.
        """
        self.cfg = cfg
        self._pretrain = False
        self._unlabel_data = None
        self._data = None
        self._len = None
        self._unlabel_len = None
        self.label = None

    def __len__(self) -> int:
        """
        return the length of the dataset, int

        In the pretrain mode, if there is any unlabeled data, return the length of all data
        In the clustering mode, return the length of the labeled data
        """
        if self._pretrain or self.label is None:
            return self.total_length
        else:
            return self.total_length - self.unlabel_length
        
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, typing.Union[torch.Tensor, None], torch.Tensor]:
        """
        return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor), 
        
        those unlabeled data should be the last to return

        for those unlabeled data, return (torch.Tensor, None, torch.Tensor)
        """
        return super().__getitem__(index)

    def pretrain(self):
        """
        set the dataset to pretrain mode, not suggested to change the default implementation
        """
        self._pretrain = True

    def clustering(self):
        """
        set the dataset to clustering mode, not suggested to change the default implementation
        """
        self._pretrain = False

    def data_init(self):
        """
        initialize the `data` of the dataset as `np.ndarray`, it should return the `np.ndarray`

        if the dataset loads each data in the __getitem__ method, the `data_init` is suggested to be implemented
        """
        return self._data

    def unlabeled_data_init(self):
        """
        initialize the `unlabel_data` of the dataset as `np.ndarray`, it should return the `np.ndarray`

        if the dataset loads each data in the __getitem__ method, the `unlabeled_data_init` is suggested to be implemented
        """
        return self._unlabel_data

    @property
    def total_length(self):
        """
        the total length of the dataset, int
       
        if the dataset loads each data in the __getitem__ method, the value of `total_length` is suggested to be set in the `__init__` method
        """
        if self._len is None and type(self._data) is torch.Tensor or type(self._data) is np.ndarray:
            self._len = self._data.shape[0]
        else:
            raise ValueError(
                "Directly set the `total_length` first or set `data` first")
        return self._len

    @total_length.setter
    def total_length(self, new_len):
        self._len = new_len

    @property
    def unlabel_length(self):
        """
        the length of the unlabel data, int, default to 0
        
        if the dataset loads each data in the __getitem__ method, the value of `unlabel_length` is suggested to be set in the `__init__` method
        """
        if self._unlabel_len is None:
            if self._unlabel_data is None:
                self._unlabel_len = 0
            elif type(self._unlabel_data) is torch.Tensor or type(self._unlabel_data) is np.ndarray:
                self._unlabel_len = self._unlabel_data.shape[0]
            else:
                raise ValueError(
                    "Directly set the `unlabel_length` first or set `unlabel_data` first")
        return self._unlabel_len

    @unlabel_length.setter
    def unlabel_length(self, new_len):
        self._unlabel_len = new_len

    @property
    def data(self) -> typing.Union[torch.Tensor, np.ndarray, None]:
        """
        all the data of the dataset containing the labeled and unlabeled data, numpy.ndarray
        
        if the dataset loads each data in the __getitem__ method, the `data_init` is suggested to be implemented
        """
        if self._data is None:
            self._data = self.data_init()
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

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
        return self._unlabel_data

    @data.setter
    def unlabel_data(self, new_data):
        self._unlabel_data = new_data

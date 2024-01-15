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

from utils import config

class ClusteringDataset(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.cfg = cfg
        self._pretrain = False
        self._unlabel_data = None
        self._data = None
        self._len = None
        self._unlabel_len = None
        self.label = None

    def __len__(self):
        if self._pretrain or self.label is None:
            return self.total_length
        else:
            return self.total_length - self.unlabel_length

    def pretrain(self):
        self._pretrain = True

    def clustering(self):
        self._pretrain = False
    
    def data_init(self):
        return self._data

    def unlabeled_data_init(self):
        return self._unlabel_data

    @property
    def total_length(self):
        if self._len is None and type(self._data) is torch.Tensor or type(self._data) is np.ndarray:
            self._len = self._data.shape[0]
        else:
            raise ValueError("Directly set the `total_length` first or set `data` first")
        return self._len

    @total_length.setter
    def total_length(self, new_len):
        self._len = new_len

    @property
    def unlabel_length(self):
        if self._unlabel_len is None:
            if self._unlabel_data is None:
                self._unlabel_len = 0
            elif type(self._unlabel_data) is torch.Tensor or type(self._unlabel_data) is np.ndarray:
                self._unlabel_len = self._unlabel_data.shape[0]
            else:
                raise ValueError("Directly set the `unlabel_length` first or set `unlabel_data` first")
        return self._unlabel_len
    
    @unlabel_length.setter
    def unlabel_length(self, new_len):
        self._unlabel_len = new_len

    @property
    def data(self):
        if self._data is None:
            self._data = self.data_init()
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
    
    @property
    def unlabel_data(self):
        if self._unlabel_data is None:
            self._unlabel_data = self.unlabeled_data_init()
        return self._unlabel_data

    @data.setter
    def unlabel_data(self, new_data):
        self._unlabel_data = new_data
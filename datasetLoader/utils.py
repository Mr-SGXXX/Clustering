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

from datasetLoader import ClusteringDataset

from .base import ClusteringDataset

class ReassignDataset(Dataset):
    def __init__(self, dataset:ClusteringDataset, new_label:typing.Union[torch.Tensor, np.ndarray, None]):
        assert type(new_label) == torch.Tensor or type(new_label) == np.ndarray, "new_label must be a torch.Tensor or np.ndarray"
        assert new_label.shape == dataset.label.shape, "new_label must have the same shape as dataset.label"
        self.dataset = dataset
        self.label = new_label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data, _, idx = self.dataset[index]
        if self.label is None:
            return data, None, idx
        else:
            return data, torch.tensor(self.label[index]), idx

def reassign_dataset(dataset: ClusteringDataset, new_label:typing.Union[torch.Tensor, np.ndarray]):
    return ReassignDataset(dataset, new_label)
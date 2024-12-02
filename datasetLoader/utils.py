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
import numpy as np
import typing
import requests
import tqdm

from .base import ClusteringDataset

class ReassignDataset(Dataset):
    """
    A dataset that reassigns the label of a ClusteringDataset

    Args:
        dataset (ClusteringDataset): The dataset to be reassigned
        new_label (typing.Union[torch.Tensor, np.ndarray]): The new label
    """
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
    """
    A function that reassigns the label of a ClusteringDataset
    """
    return ReassignDataset(dataset, new_label)

def download_dataset(url:str, save_path:str):
    """
    A function that downloads a dataset from a URL and saves it to a file

    Args:
        url (str): The URL of the dataset
        save_path (str): The path to save the dataset
    """
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with open(save_path, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for data in r.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
    return save_path
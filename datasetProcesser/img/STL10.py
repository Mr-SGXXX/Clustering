import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import os

from utils import config

class STL10(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'STL10'
        data_dir = cfg.get("global", "dataset_dir")
        train_dataset = datasets.STL10(data_dir, split='train', download=True)
        test_dataset = datasets.STL10(data_dir, split='test', download=True)
        self.data = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        if 'img' in needed_data_types:
            self.data_type = 'img'
            self.input_dim = self.data.shape[1:]
        elif 'seq' in needed_data_types:
            self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32)
            self.data_type = 'seq'
            self.input_dim = self.data.shape[1]
        else:
            raise ValueError(f"Not available data type for STL10 in {needed_data_types}")
        self.label = np.concatenate((train_dataset.labels, test_dataset.labels), axis=0)
        self.label = self.label.reshape((self.label.size,))

        self.num_classes = len(np.unique(self.label))

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index])), \
            torch.from_numpy(np.array(self.label[index])), \
            torch.from_numpy(np.array(index))
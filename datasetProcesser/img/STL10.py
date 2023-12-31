import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import os

from .utils import ResNet50Extractor
from utils import config

class STL10(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'STL10'
        data_dir = cfg.get("global", "dataset_dir")
        train_dataset = datasets.STL10(data_dir, split='train', download=True)
        unlabel_dataset = datasets.STL10(data_dir, split='unlabeled', download=True)
        test_dataset = datasets.STL10(data_dir, split='test', download=True)
        self.data = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        self.unlabel_data = unlabel_dataset.data
        if 'img' in needed_data_types:
            self.data_type = 'img'
            self.input_dim = self.data.shape[1:]
        elif 'seq' in needed_data_types:
            if cfg.get("STL10", "img2seq_method") == 'flatten':
                self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32)
            elif cfg.get("STL10", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'stl10_binary')
                if os.path.exists(os.path.join(data_dir, 'STL10_resnet50.npy')):
                    self.data = np.load(os.path.join(data_dir, 'STL10_resnet50.npy'))
                else:
                    self.data = ResNet50Extractor(self.data, cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_resnet50.npy'), self.data)
            elif cfg.get("STL10", "img2seq_method") == 'hog':
                pass
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
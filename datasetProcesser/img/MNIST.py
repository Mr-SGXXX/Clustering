import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import os

from .utils import ResNet50Extractor
from utils import config

class MNIST(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'MNIST'
        data_dir = cfg.get("global", "dataset_dir")
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True)
        self.data = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        self.data = self.data.reshape((self.data.shape[0], 1, self.data.shape[1], self.data.shape[2]))
        if 'img' in needed_data_types:
            self.data_type = 'img'
            self.input_dim = self.data.shape[1:]
        elif 'seq' in needed_data_types:
            if cfg.get("MNIST", "img2seq_method") == 'flatten':
                self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32) * 0.02
            elif cfg.get("MNIST", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'MNIST')
                if os.path.exists(os.path.join(data_dir, 'MNIST_resnet50.npy')):
                    self.data = np.load(os.path.join(data_dir, 'MNIST_resnet50.npy'))
                else:
                    self.data = ResNet50Extractor(self.data, cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'MNIST_resnet50.npy'), self.data)
            self.data_type = 'seq'
            self.input_dim = self.data.shape[1]
        else:
            raise ValueError(f"Not available data type for MNIST in {needed_data_types}")
        self.label = np.concatenate((train_dataset.targets, test_dataset.targets), axis=0)
        self.label = self.label.reshape((self.label.size,))
        
        self.num_classes = len(np.unique(self.label))
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index])), \
            torch.from_numpy(np.array(self.label[index])), \
            torch.from_numpy(np.array(index))

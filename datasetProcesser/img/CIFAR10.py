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
import torchvision.datasets as datasets
import numpy as np
import os

from .utils import ResNet50Extractor
from utils import config

class CIFAR10(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'CIFAR10'
        data_dir = cfg.get("global", "dataset_dir")
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True)
        self.data = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        self.data = self.data.transpose((0, 3, 1, 2))
        self.unlabel_data = None
        if 'img' in needed_data_types:
            self.data_type = 'img'
            self.input_dim = self.data.shape[1:]
        elif 'seq' in needed_data_types:
            if cfg.get("CIFAR10", "img2seq_method") == 'flatten':
                self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32)
            elif cfg.get("CIFAR10", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
                if os.path.exists(os.path.join(data_dir, 'CIFAR10_resnet50.npy')):
                    self.data = np.load(os.path.join(data_dir, 'CIFAR10_resnet50.npy'))
                else:
                    self.data = ResNet50Extractor(self.data, cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'CIFAR10_resnet50.npy'), self.data)
            else:
                raise ValueError(f"`{cfg.get('CIFAR10', 'img2seq_method')}` is not an available img2seq_method for CIFAR10")
            self.data_type = 'seq'
            self.input_dim = self.data.shape[1]
        else:
            raise ValueError(f"No available data type for CIFAR10 in {needed_data_types}")
        self.label = np.concatenate((train_dataset.targets, test_dataset.targets), axis=0)
        self.label = self.label.reshape((self.label.size,))

        self.num_classes = len(np.unique(self.label))

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index])), \
            torch.from_numpy(np.array(self.label[index])), \
            torch.from_numpy(np.array(index))
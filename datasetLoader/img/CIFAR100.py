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

# the sparse2coarse function refers to https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
import torch
import torchvision.datasets as datasets
import numpy as np
import os

from ..base import ClusteringDataset
from .utils import ResNet50Extractor
from utils import config

class CIFAR100(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        super().__init__(cfg, needed_data_types)
    
    def label_data_init(self):
        super_class_flag = self.cfg.get("CIFAR100", "super_class")
        train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True)
        if super_class_flag:
            train_dataset.targets = sparse2coarse(train_dataset.targets)
            test_dataset.targets = sparse2coarse(test_dataset.targets)
        data = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        data = data.transpose((0, 3, 1, 2))
        if 'img' in self.needed_data_types:
            self.data_type = 'img'
            self.name += '_img'
        elif 'seq' in self.needed_data_types:
            if self.cfg.get("CIFAR100", "img2seq_method") == 'flatten':
                data = data.reshape(data.shape[0], -1).astype(np.float32)
                self.name += '_seq_flatten'
            elif self.cfg.get("CIFAR100", "img2seq_method") == 'resnet50':
                if os.path.exists(os.path.join(self.data_dir, 'CIFAR100_resnet50.npy')):
                    data = np.load(os.path.join(self.data_dir, 'CIFAR100_resnet50.npy'))
                else:
                    data = ResNet50Extractor(data, self.cfg)().astype(np.float32)
                    np.save(os.path.join(self.data_dir, 'CIFAR100_resnet50.npy'), data)
                self.name += '_seq_resnet50'
            else:
                raise ValueError(f"`{self.cfg.get('CIFAR100', 'img2seq_method')}` is not an available img2seq_method for CIFAR100")
            self.data_type = 'seq'
        else:
            raise ValueError(f"No available data type for CIFAR100 in {self.needed_data_types}")
        label = np.concatenate((train_dataset.targets, test_dataset.targets), axis=0)
        label = label.reshape((label.size,))
        return data, label

    def data_preprocess(self, sample):
        return sample
    

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]
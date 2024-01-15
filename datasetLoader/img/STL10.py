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
import torchvision.datasets as datasets
import numpy as np
import os 

from ..base import ClusteringDataset
from .utils import ResNet50Extractor, extract_hog_features, extract_color_map_features
from utils import config

class STL10(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'STL10'
        data_dir = cfg.get("global", "dataset_dir")
        train_dataset = datasets.STL10(data_dir, split='train', download=True)
        unlabel_dataset = datasets.STL10(data_dir, split='unlabeled', download=True)
        test_dataset = datasets.STL10(data_dir, split='test', download=True)
        self.label_data = np.concatenate((train_dataset.data.numpy(), test_dataset.data.numpy()), axis=0)
        self.unlabel_data = unlabel_dataset.data.numpy()
        self.data = np.concatenate((self.label_data, self.unlabel_data), axis=0)
        if 'img' in needed_data_types:
            self.data_type = 'img'
            self.input_dim = self.data.shape[1:]
        elif 'seq' in needed_data_types:
            if cfg.get("STL10", "img2seq_method") == 'flatten':
                self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32)
                self.unlabel_data = self.unlabel_data.reshape(self.unlabel_data.shape[0], -1).astype(np.float32)
            elif cfg.get("STL10", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'stl10_binary')
                if os.path.exists(os.path.join(data_dir, 'STL10_resnet50.npy')):
                    self.data = np.load(os.path.join(data_dir, 'STL10_resnet50.npy'))
                else:
                    self.data = ResNet50Extractor(self.data, cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_resnet50.npy'), self.data)
                if os.path.exists(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy')):
                    self.unlabel_data = np.load(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy'))
                else:
                    self.unlabel_data = ResNet50Extractor(self.unlabel_data, cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy'), self.unlabel_data)
            elif cfg.get("STL10", "img2seq_method") == 'hog_color':
                if os.path.exists(os.path.join(data_dir, 'STL10_hog_color.npy')):
                    self.data = np.load(os.path.join(data_dir, 'STL10_hog_color.npy'))
                else:
                    self.data = self.data.transpose((0, 2, 3, 1))
                    features = [
                        np.concatenate((extract_hog_features(img), extract_color_map_features(img)), axis=0) for img in self.data
                    ]
                    self.data = np.array(features).astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_hog_color.npy'), self.data)
            else:
                raise ValueError(f"`{cfg.get('STL10', 'img2seq_method')}` is not an available img2seq_method for STL10")
            self.data_type = 'seq'
            self.input_dim = self.data.shape[1]
        else:
            raise ValueError(f"No available data type for STL10 in {needed_data_types}")
        self.label = np.concatenate((train_dataset.labels, test_dataset.labels), axis=0)
        self.label = self.label.reshape((self.label.size,))

        self.num_classes = len(np.unique(self.label))
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index])), \
            torch.from_numpy(np.array(self.label[index])), \
            torch.from_numpy(np.array(index))
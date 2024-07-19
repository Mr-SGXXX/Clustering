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
import torchvision.datasets as datasets
import numpy as np
import os 

from ..base import ClusteringDataset
from .utils import ResNet50Extractor, extract_hog_features, extract_color_map_features
from utils import config

class STL10(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        super().__init__(cfg, needed_data_types)
        self.name = 'STL10'
        
    def label_data_init(self):
        data_dir = self.cfg.get("global", "dataset_dir")
        train_dataset = datasets.STL10(data_dir, split='train', download=True)
        test_dataset = datasets.STL10(data_dir, split='test', download=True)
        data = np.concatenate((train_dataset.data.numpy(), test_dataset.data.numpy()), axis=0)
        if 'img' in self.needed_data_types:
            self.data_type = 'img'
        elif 'seq' in self.needed_data_types:
            if self.cfg.get("STL10", "img2seq_method") == 'flatten':
                data = data.reshape(data.shape[0], -1).astype(np.float32)
            elif self.cfg.get("STL10", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'stl10_binary')
                if os.path.exists(os.path.join(data_dir, 'STL10_label_resnet50.npy')):
                    data = np.load(os.path.join(data_dir, 'STL10_label_resnet50.npy'))
                else:
                    data = ResNet50Extractor(data, self.cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_label_resnet50.npy'), data)
            elif self.cfg.get("STL10", "img2seq_method") == 'hog_color':
                if os.path.exists(os.path.join(data_dir, 'STL10_label_hog_color.npy')):
                    data = np.load(os.path.join(data_dir, 'STL10_label_hog_color.npy'))
                else:
                    data = data.transpose((0, 2, 3, 1))
                    features = [
                        np.concatenate((extract_hog_features(img), extract_color_map_features(img)), axis=0) for img in data
                    ]
                    data = np.array(features).astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_hog_color.npy'), data)
            else:
                raise ValueError(f"`{self.cfg.get('STL10', 'img2seq_method')}` is not an available img2seq_method for STL10")
            self.data_type = 'seq'
        else:
            raise ValueError(f"No available data type for STL10 in {self.needed_data_types}")
        label = np.concatenate((train_dataset.labels, test_dataset.labels), axis=0)
        label = label.reshape((label.size,))
        return data, label

    def unlabeled_data_init(self):
        data_dir = self.cfg.get("global", "dataset_dir")
        unlabel_dataset = datasets.STL10(data_dir, split='unlabeled', download=True)
        data = unlabel_dataset.data.numpy()
        if 'img' in self.needed_data_types:
            pass
        elif 'seq' in self.needed_data_types:
            if self.cfg.get("STL10", "img2seq_method") == 'flatten':
                data = data.reshape(data.shape[0], -1).astype(np.float32)
            elif self.cfg.get("STL10", "img2seq_method") == 'resnet50':
                data_dir = os.path.join(data_dir, 'stl10_binary')
                if os.path.exists(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy')):
                    data = np.load(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy'))
                else:
                    data = ResNet50Extractor(data, self.cfg)().astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_unlabel_resnet50.npy'), data)
            elif self.cfg.get("STL10", "img2seq_method") == 'hog_color':
                if os.path.exists(os.path.join(data_dir, 'STL10_unlabel_hog_color.npy')):
                    data = np.load(os.path.join(data_dir, 'STL10_unlabel_hog_color.npy'))
                else:
                    data = data.transpose((0, 2, 3, 1))
                    features = [
                        np.concatenate((extract_hog_features(img), extract_color_map_features(img)), axis=0) for img in data
                    ]
                    data = np.array(features).astype(np.float32)
                    np.save(os.path.join(data_dir, 'STL10_hog_color.npy'), data)
            else:
                raise ValueError(f"`{self.cfg.get('STL10', 'img2seq_method')}` is not an available img2seq_method for STL10")
        else:
            raise ValueError(f"No available data type for STL10 in {self.needed_data_types}")
        return data
    
    def data_preprocess(self, sample):
        return sample

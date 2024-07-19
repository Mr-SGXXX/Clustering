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
import numpy as np
import os

from ..base import ClusteringDataset
from utils import config

class Reuters10K(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        super().__init__(cfg, needed_data_types)
        self.name = 'Reuters10K'

    def label_data_init(self):
        if 'seq' not in self.needed_data_types:
            raise ValueError(f"Not available data type for reuters10k in {self.needed_data_types}")
        self.data_type = 'seq'
        data_dir = os.path.join(self.cfg.get("global", "dataset_dir"), "Reuters10K")
        data_path = os.path.join(data_dir, "reutersidf10k.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found")
        data = np.load(data_path, allow_pickle=True).item()
        X = data['data']
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        Y = data['label']
        Y = Y.reshape((Y.size,))
        return X, Y
    
    def data_preprocess(self, sample):
        return sample

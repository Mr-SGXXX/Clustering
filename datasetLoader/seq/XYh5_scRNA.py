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

# This dataset loader is for loading scRNA data in h5 format
import torch
import numpy as np
import os
import h5py

from ..base import ClusteringDataset
from .utils import load_scRNA_h5data
from utils import config

class XYh5_scRNA(ClusteringDataset):
    def __init__(self, cfg: config, needed_data_types:list):
        super().__init__(cfg, needed_data_types)
        self.name = None

    def label_data_init(self):
        if 'seq' not in self.needed_data_types:
            raise ValueError(f"No available data type for XYh5_scRNA in {self.needed_data_types}")
        self.data_type = 'seq'
        data_name = self.cfg.get("XYh5_scRNA", "data_name")
        data_dir = os.path.join(self.cfg.get("global", "dataset_dir"), "XYh5_scRNA")
        data_path = os.path.join(data_dir, f"{data_name}.h5")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found")
        X, Y = load_scRNA_h5data(data_path, self.cfg)
        return X, Y
    
    def data_preprocess(self, sample) -> torch.Tensor | np.ndarray:
        return sample
        
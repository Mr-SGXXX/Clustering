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
import numpy as np
import os

from utils import config

class Reuters10K(Dataset):
    def __init__(self, cfg: config, needed_data_types:list):
        self.name = 'Reuters10K'
        if 'seq' not in needed_data_types:
            raise ValueError(f"Not available data type for reuters10k in {needed_data_types}")
        self.data_type = 'seq'
        data_dir = os.path.join(cfg.get("global", "dataset_dir"), "Reuters10K")
        data_path = os.path.join(data_dir, "reutersidf10k.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found")
        data = np.load(data_path, allow_pickle=True).item()
        self.data = data['data']
        self.data = self.data.reshape(self.data.shape[0], -1).astype(np.float32)
        self.unlabel_data = None
        self.label = data['label']
        self.label = self.label.reshape((self.label.size,))
        self.input_dim = self.data.shape[1]
        self.num_classes = len(np.unique(self.label))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data[index])), \
            torch.from_numpy(np.array(self.label[index])), \
            torch.from_numpy(np.array(index))
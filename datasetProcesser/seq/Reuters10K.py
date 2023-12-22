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
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
from logging import Logger

from datasetLoader import ClusteringDataset
from utils import config


class ClassicalMethod:
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        self.dataset = dataset
        self.cfg = cfg
        self.description = description
        self.logger = logger
        self.device = cfg.get("global", "device")
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            self.n_clusters = dataset.num_classes
        else:
            self.n_clusters = cfg.get("global", "n_clusters")
            assert type(
                self.n_clusters) is int, "n_clusters should be of type int"
            assert self.n_clusters > 0, "n_clusters should be larger than 0"
    
    
    def fit(self):
        raise NotImplementedError
        # this method should return the predicted labels and the features clustered.

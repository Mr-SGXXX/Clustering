from utils import config
from logging import Logger

class ClassicalMethod:
    def __init__(self, dataset, description, logger:Logger, cfg: config):
        self.dataset = dataset
        self.cfg = cfg
        self.description = description
        self.logger = logger
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

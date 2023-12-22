from utils import config

class ClassicalMethod:
    def __init__(self, cfg: config):
        self.cfg = cfg
    
    def fit(self, data):
        raise NotImplementedError

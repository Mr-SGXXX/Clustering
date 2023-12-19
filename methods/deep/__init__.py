from EDESC import EDESC

"""
The Dictionary of Deep clustering methods

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, logger, cfg)`, `pretrain()`, `train()` and `forward()` methods.
train() should return the predicted labels, features and metrics.
"""
DEEP_METHODS = {
    "EDESC": EDESC,
}

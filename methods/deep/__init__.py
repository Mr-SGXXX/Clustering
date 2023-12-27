from .EDESC import EDESC
from .DEC import DEC
from .IDEC import IDEC

"""
The Dictionary of Deep clustering methods

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, logger, cfg)`, `pretrain()`, `train()` and `forward()` methods.
pretrain() should return the hidden layer representation of the whole dataset and the pretrain loss list.
train_model() should return the predicted labels, features and metrics. There must be a `total_loss` update in the metrics.
"""
DEEP_METHODS = {
    # Cai J, Fan J, Guo W, et al. Efficient deep embedded subspace clustering
    # Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1-10.
    # Link: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.html
    "EDESC": EDESC,
    "DEC": DEC,
    "IDEC": IDEC,
}

"""
DEEP_METHODS_INPUT_IMG_FLAG: A dict of flags indicating whether the input data is an image.

The key is the name of the method, and the value is the list of what type of input data the method can accept
such as "seq" for sequential data, "img" for image data, "tab" for tabular data, "graph" for graph data, etc
"""
DEEP_METHODS_INPUT_TYPES = {
    "EDESC": ["seq", "img"],
    "DEC": ["seq", "img"],
    "IDEC": ["seq", "img"],
}

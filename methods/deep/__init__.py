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
    # CVPR. 2022: 1-10.
    # Link: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.html
    "EDESC": EDESC,
    # Xie J, Girshick R, Farhadi A. Unsupervised deep embedding for clustering analysis
    # ICML. 2016, 478-487.
    # Link: http://proceedings.mlr.press/v48/xieb16.pdf
    "DEC": DEC,
    # Guo X, Gao L, Liu X, et al. Improved deep embedded clustering with local structure preservation
    # IJCAI. 2017, 17: 1753-1759.
    # Link: https://www.ijcai.org/proceedings/2017/0243.pdf
    "IDEC": IDEC,
}

"""
DEEP_METHODS_INPUT_IMG_FLAG: A dict of flags indicating whether the input data is an image.

The key is the name of the method, and the value is the list of what type of input data the method can directly accept
such as "seq" for sequential data, "img" for image data, "tab" for tabular data, "graph" for graph data, etc
"""
DEEP_METHODS_INPUT_TYPES = {
    "EDESC": ["seq"],
    "DEC": ["seq"],
    "IDEC": ["seq"],
}

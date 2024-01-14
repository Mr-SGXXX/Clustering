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
from .EDESC import EDESC
from .DEC import DEC
from .IDEC import IDEC

"""
The Dictionary of Deep clustering methods

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, description, logger, cfg)`, `pretrain()`, `train()` and `forward(...)` methods.
pretrain() should return the hidden layer representation of the whole dataset.
There must be any loss update in the metrics by `update_pretrain_loss` to draw the pretrain loss figure.
For those methods that do not need pretrain, just return None.

train_model() should return the predicted labels, features. 
There must be a `total_loss` meaning the total model loss update in the metrics by `update_loss` to draw the clustering loss figure.
If ground truth is available, the `y_true` should be passed to the `update` method of the metrics.
the args must be the same as the base class in `base.py`.
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
DEEP_METHODS_INPUT_TYPES: The Dictionary of Deep clustering methods' input types

The key is the name of the method, and the value is the list of what type of input data the method can directly accept
such as "seq" for sequential data, "img" for image data, "graph" for graph data, etc
"""
DEEP_METHODS_INPUT_TYPES = {
    "EDESC": ["seq"],
    "DEC": ["seq"],
    "IDEC": ["seq"],
}

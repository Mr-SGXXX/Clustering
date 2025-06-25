# MIT License

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

from .EDESC import EDESC
from .DEC import DEC
from .IDEC import IDEC
from .DeepCluster import DeepCluster
from .CC import CC
from .DivClust import DivClust
from .IIC import IIC
from .graphs.node2vec import node2vec
from .graphs.GRACE_2017 import GRACE as GRACE_2017
from .graphs.DiffPool import DiffPool
from .graphs.DGI import DGI
from .graphs.GRACE_2020 import GRACE as GRACE_2020
from .graphs.MVGRL import MVGRL
from .graphs.SDCN import SDCN
from .graphs.DFCN import DFCN
from .graphs.S3GC import S3GC
from .graphs.MAGI import MAGI
from .graphs.DGCluster import DGCluster
from .graphs.DCRN import DCRN
from .graphs.DMoN import DMoN
from .graphs.MinCutPool import MinCutPool
from .graphs.MGCN import MGCN
from .graphs.GraphMAE import GraphMAE

"""
The Dictionary of Deep clustering methods

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, description, logger, cfg)`, `pretrain()`, `train()` and `forward(...)` methods.

forward() is the forward propagation of the model for convinience. If you don't use it, you can just leave it aside.

pretrain() should return the hidden layer representation of the whole dataset.
There must be any loss update in the metrics by `update_pretrain_loss` to draw the pretrain loss figure.
For those methods that do not need pretrain, just return None.

clustering() should return the predicted labels, features. 
There must be a `total_loss` meaning the total model loss update in the metrics by `update_loss` to draw the clustering loss figure.
If ground truth is available, the `y_true` should be passed to the `update` method of the metrics.
the args must be the same as the base class in `base.py`.
"""
DEEP_METHODS = {
    # Xie J, Girshick R, Farhadi A. Unsupervised deep embedding for clustering analysis
    # ICML. 2016, 478-487.
    # Link: http://proceedings.mlr.press/v48/xieb16.pdf
    "DEC": DEC,
    # Guo X, Gao L, Liu X, et al. Improved deep embedded clustering with local structure preservation
    # IJCAI. 2017, 17: 1753-1759.
    # Link: https://www.ijcai.org/proceedings/2017/0243.pdf
    "IDEC": IDEC,
    # Cai J, Fan J, Guo W, et al. Efficient deep embedded subspace clustering
    # CVPR. 2022: 1-10.
    # Link: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.html
    "EDESC": EDESC,
    "DivClust": DivClust,
    "IIC": IIC,
    "CC": CC,
    "DeepCluster": DeepCluster,
    
    
    # Graph Clustering methods
    # Grover A, Leskovec J.
    # node2vec: Scalable feature learning for networks[C]
    # Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016: 855-864.
    # https://dl.acm.org/doi/abs/10.1145/2939672.2939754
    "node2vec": node2vec,
    # Yang C, Liu M, Wang Z, et al.
    # Graph clustering with dynamic embedding[J].
    # arXiv preprint arXiv:1712.08249, 2017.
    # https://arxiv.org/abs/1712.08249
    "GRACE_2017": GRACE_2017,
    # Ying Z, You J, Morris C, et al.
    # Hierarchical graph representation learning with differentiable pooling[J].
    # Advances in neural information processing systems, 2018, 31.
    # https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html
    "DiffPool": DiffPool,    
    # Veličković P, Fedus W, Hamilton W L, et al.
    # Deep graph infomax[J].
    # arXiv preprint arXiv:1809.10341, 2018.
    # https://arxiv.org/abs/1809.10341
    "DGI": DGI,
    # Zhu Y, Xu Y, Yu F, et al.
    # Deep graph contrastive representation learning[J].
    # arXiv preprint arXiv:2006.04131, 2020.
    # https://arxiv.org/abs/2006.04131
    "GRACE_2020": GRACE_2020,
    # Hassani K, Khasahmadi A H. 
    # Contrastive multi-view representation learning on graphs[C]
    # International conference on machine learning. PMLR, 2020: 4116-4126.
    # https://proceedings.mlr.press/v119/hassani20a.html
    "MVGRL": MVGRL,
    # Bo D, Wang X, Shi C, et al.
    # Structural deep clustering network
    # Proceedings of the web conference 2020. 2020: 1400-1410.
    # https://dl.acm.org/doi/abs/10.1145/3366423.3380214
    "SDCN": SDCN,
    # Bianchi F M, Grattarola D, Alippi C.
    # Spectral clustering with graph neural networks for graph pooling[C]
    # International conference on machine learning. PMLR, 2020: 874-883.
    # https://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf
    "MinCutPool": MinCutPool,
    # Tu W, Zhou S, Liu X, et al. 
    # Deep fusion clustering network[C]
    # Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(11): 9978-9987.
    # https://ojs.aaai.org/index.php/AAAI/article/view/17198/17005
    "DFCN": DFCN,
    # Liu Y, Tu W, Zhou S, et al. 
    # Deep graph clustering via dual correlation reduction[C]
    # Proceedings of the AAAI conference on artificial intelligence. 2022, 36(7): 7603-7611.
    # https://ojs.aaai.org/index.php/AAAI/article/view/20726/20485
    "DCRN": DCRN,
    # Devvrit F, Sinha A, Dhillon I, et al.
    # S3GC: scalable self-supervised graph clustering[J].
    # Advances in Neural Information Processing Systems, 2022, 35: 3248-3261.
    # https://proceedings.neurips.cc/paper_files/paper/2022/file/15972a9575e0f03bf82f00aebeb40774-Paper-Conference.pdf
    "S3GC": S3GC,
    # Tsitsulin A, Palowitch J, Perozzi B, et al.
    # Graph clustering with graph neural networks[J].
    # Journal of Machine Learning Research, 2023, 24(127): 1-21.
    # https://www.jmlr.org/papers/volume24/20-998/20-998.pdf
    "DMoN": DMoN,
    # Li X, Wu W, Zhang B, et al.
    # Multi-scale Graph Clustering Network[J].
    # Information Sciences, 2024: 121023.
    # https://www.sciencedirect.com/science/article/pii/S002002552400937X
    "MGCN": MGCN,
    # Bhowmick A, Kosan M, Huang Z, et al. 
    # DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization
    # Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(10): 11069-11077.
    # https://ojs.aaai.org/index.php/AAAI/article/view/28983/29868
    "DGCluster": DGCluster,
    # Liu Y, Li J, Chen Y, et al. 
    # Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective[C]
    # Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024: 1968-1979.
    # https://dl.acm.org/doi/pdf/10.1145/3637528.3671967
    "MAGI": MAGI,
    # Hou Z, Liu X, Cen Y, et al. 
    # Graphmae: Self-supervised masked graph autoencoders[C]
    # Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022: 594-604.
    # https://dl.acm.org/doi/pdf/10.1145/3534678.3539321
    "GraphMAE": GraphMAE,
}

"""
DEEP_METHODS_INPUT_TYPES: The Dictionary of Deep clustering methods' input types

The key is the name of the method, and the value is the list of what type of input data the method can directly accept
such as "seq" for sequential data, "img" for image data, etc
"""
DEEP_METHODS_INPUT_TYPES = {
    "EDESC": ["seq"],
    "DeepCluster": ["img"],
    "DEC": ["seq"],
    "IDEC": ["seq"],
    
    # Graph Clustering methods
    "node2vec": ["seq"],
    "GRACE_2017": ["seq"],
    "DiffPool": ["seq"],
    "DGI": ["seq"],
    "GRACE_2020": ["seq"],
    "MVGRL": ["seq"],
    "SDCN": ["seq"],
    "MinCutPool": ["seq"],
    "DFCN": ["seq"],
    "DCRN": ["seq"],
    "DMoN": ["seq"],
    "S3GC": ["seq"],
    "MGCN": ["seq"],
    "DGCluster": ["seq"],
    "MAGI": ["seq"],
    "GraphMAE": ["seq"],
}

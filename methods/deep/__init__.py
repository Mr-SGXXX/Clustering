from EDESC import EDESC

"""
The Dictionary of Deep clustering methods

The key is the name of the method, and the value is the class of the method.
Each method should consist of `__init__(dataset, logger, cfg)`, `pretrain()`, `train()` and `forward()` methods.
train() should return the predicted labels, features and metrics.
"""
DEEP_METHODS = {
    # Cai J, Fan J, Guo W, et al. Efficient deep embedded subspace clustering
    # Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1-10.
    # Link: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.html
    "EDESC": EDESC,
}

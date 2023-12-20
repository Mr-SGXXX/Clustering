from .Reuters10K import Reuters10K
from .MNIST import MNIST

# Each dataset here should be a `torch.utils.Dataset` class with following elements:
#   label: the label of the dataset, if label is not available, set it to None, numpy.ndarray
#   data: the data of the dataset, numpy.ndarray
#   input_dim: the input dimension of the dataset
#   num_classes: the number of classes of the dataset
#   output_img_flag: whether the dataset outputs a image 
# The methods of the dataset class:
#   __init__: initialize the dataset, need a config object as input and a flag indicating whether the dataset needs to output a image, this feature is used for some methods that specially designed for image datasets
# when the dataset is not an image dataset, the output_img_flag should be set to False
#   __len__: return the length of the dataset
#   __getitem__: return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor)
# if the output_img_flag is True, the output data should be a torch.Tensor with shape (C, H, W), otherwise, the data should be a torch.Tensor with shape (D, )


"""
DATASETS: The dict of datasets.

The key is the name of the dataset, and the value is the class of the dataset.
"""
DATASETS = {
    "Reuters10K": Reuters10K,
    "MNIST": MNIST
}
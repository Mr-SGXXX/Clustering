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

from .img import IMG_DATASETS
from .seq import SEQ_DATASETS
from .base import ClusteringDataset
from .seq.utils import reassign_dataset

# Each dataset here should be a `ClusteringDataset` class with following elements:
#   name: the name of the dataset, str
#   label: the label of the dataset, if label is not available, set it to None, numpy.ndarray, default to None
#   data: the data of the dataset containing the labeled and unlabeled data, numpy.ndarray
#         if the dataset loads each data in the __getitem__ method, the `data_init` is suggested to be implemented
#   unlabel_data: the unlabel data of the dataset, if no unlabel data, set it to None, numpy.ndarray, default to None
#                 commonly, the unlabel data is only used in pretraining.
#                 remember that the unlabel data must be the last part in the self.data
#   total_length: the total length of the dataset, int
#                 if the dataset loads each data in the __getitem__ method, the value of `total_length` is suggested to be set in the `__init__` method
#   unlabel_length: the length of the unlabel data, int, default to 0
#                   if the dataset loads each data in the __getitem__ method, the value of `unlabel_length` is suggested to be set in the `__init__` method
#   data_type: the type of the data, should be one of ['seq', 'img', ...], str
#   input_dim: the input dimension of the dataset
#   num_classes: the number of classes of the dataset, int

# The methods of the dataset class:
#   __init__: initialize the dataset, need a config object as input and a flag list indicating what data type the dataset can accept
#   __getitem__: return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor), 
#                if the label is not available or those unlabeled data, return (torch.Tensor, None, torch.Tensor)
#  data_init(): initialize the `data` of the dataset as `np.ndarray`, it should return the `np.ndarray`
#  unlabeled_data_init(): initialize the `unlabel_data` of the dataset as `np.ndarray`, it should return the `np.ndarray`
#  pretrain(): set the dataset to pretrain mode, not suggested to change the default implementation
#  clustering(): set the dataset to clustering mode, not suggested to change the default implementation
#  __len__: return the length of the dataset, int, no suggesting to change the default implementation


"""
DATASETS: The dict of datasets.

The key is the name of the dataset, and the value is the class of the dataset.
"""
DATASETS = {
    **IMG_DATASETS,
    **SEQ_DATASETS,
}
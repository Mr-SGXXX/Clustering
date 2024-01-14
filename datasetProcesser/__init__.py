from .img import IMG_DATASETS
from .seq import SEQ_DATASETS


# Each dataset here should be a `torch.utils.Dataset` class with following elements:
#   name: the name of the dataset, str
#   label: the label of the dataset, if label is not available, set it to None, numpy.ndarray
#   data: the data of the dataset, numpy.ndarray
#         if the dataset loads each data in the __getitem__ method, the data is suggest to be implemented as a property decorator
#   unlabel_data: the unlabel data of the dataset, if no unlabel data, set it to None, numpy.ndarray
#                 commomly, the unlabel data is only used in pretraining.
#   data_type: the type of the data, should be one of ['seq', 'img', ...], str
#   input_dim: the input dimension of the dataset
#   num_classes: the number of classes of the dataset, it must

# The methods of the dataset class:
#   __init__: initialize the dataset, need a config object as input and a flag list indicating what data type the dataset can accept
#   __len__: return the length of the dataset
#   __getitem__: return the data, label and index of the dataset, (torch.Tensor, torch.Tensor, torch.Tensor)
# if the output_img_flag is True, the output data should be a torch.Tensor with shape (C, H, W), otherwise, the data should be a torch.Tensor with shape (D, )


"""
DATASETS: The dict of datasets.

The key is the name of the dataset, and the value is the class of the dataset.
"""
DATASETS = {
    **IMG_DATASETS,
    **SEQ_DATASETS,
}
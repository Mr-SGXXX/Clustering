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
from .MNIST import MNIST
from .FashionMNIST import FashionMNIST
from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .STL10 import STL10
from .USPS import USPS
from .ImageNet_Dogs import ImageNet_Dogs
from .ImageNet_10 import ImageNet_10

IMG_DATASETS = {
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "STL10": STL10,
    "USPS": USPS,
    "ImageNet-Dogs": ImageNet_Dogs,
    "ImageNet-10": ImageNet_10,
}
from .MNIST import MNIST
from .FashionMNIST import FashionMNIST
from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .STL10 import STL10

IMG_DATASETS = {
    "MNIST": MNIST,
    "FashionMNIST": FashionMNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "STL10": STL10,
}
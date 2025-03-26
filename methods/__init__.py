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
import os

from .deep import *
from .classical import *

METHODS_INPUT_TYPES = {
    **DEEP_METHODS_INPUT_TYPES,
    **CLASSICAL_METHODS_INPUT_TYPES
}


"""
    If you want to add your own methods, you can create the file methods/demo/__init__.py and add your methods in it referring to the following format:
    MY_METHODS = {
        "Your Method Name" : your_method_class
    }

    MY_METHODS_TYPE_FLAG = {
        "Your Method Name" : "deep" or "classical"
    }

    MY_METHODS_INPUT_TYPES = {
        “Your Method Name” : ["input_type1 (seq)", "input_type2 (img)", ...]
    }
"""

from .demo import *
for method in MY_METHODS:
    if MY_METHODS_TYPE_FLAG[method] == "deep":
        DEEP_METHODS[method] = MY_METHODS[method]
    elif MY_METHODS_TYPE_FLAG[method] == "classical":
        CLASSICAL_METHODS[method] = MY_METHODS[method]
    else:
        raise ValueError("The method type should be either 'deep' or 'classical'")
METHODS_INPUT_TYPES = {**METHODS_INPUT_TYPES, **MY_METHODS_INPUT_TYPES}


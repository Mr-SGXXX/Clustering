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

# try:
from .Graph1 import Graph1, Graph2, Graph3, Graph4
# from .Graph1 import Graph2 as Graph1

MY_METHODS = {
    "Graph1" : Graph1,
    "Graph2" : Graph2,
    "Graph3" : Graph3,
    "Graph4" : Graph4
}

MY_METHODS_TYPE_FLAG = {
    "Graph1" : "deep",
    "Graph2" : "deep",
    "Graph3" : "deep",
    "Graph4" : "deep"
}

MY_METHODS_INPUT_TYPES = {
    "Graph1" : ["img", "seq"],
    "Graph2" : ["img", "seq"],
    "Graph3" : ["img", "seq"],
    "Graph4" : ["img", "seq"]
}
# except:
#     MY_METHODS = {}
#     MY_METHODS_TYPE_FLAG = {}
#     MY_METHODS_INPUT_TYPES = {}
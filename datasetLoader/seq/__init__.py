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
from .Reuters10K import Reuters10K
from .XYh5_scRNA import XYh5_scRNA
from .ACM import ACM
from .DBLP import DBLP
from .Cora import Cora
from .Citeseer import Citeseer
from .Pubmed import Pubmed
from .Wiki import Wiki
from .AMAP import AMAP

SEQ_DATASETS = {
    "Reuters10K": Reuters10K,
    "XYh5_scRNA": XYh5_scRNA,
    "ACM": ACM,
    "DBLP": DBLP,
    "Cora": Cora,
    "Citeseer": Citeseer,
    "Pubmed": Pubmed,
    "Wiki": Wiki,
    "AMAP": AMAP
}
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
from .Reuters10K import Reuters10K
from .XYh5_scRNA import XYh5_scRNA
from .graphs.ACM import ACM
from .graphs.DBLP import DBLP
from .graphs.Cora import Cora
from .graphs.Citeseer import Citeseer
from .graphs.Pubmed import Pubmed
from .graphs.Wiki import Wiki
from .graphs.BAT import BAT
from .graphs.EAT import EAT
from .graphs.UAT import UAT
from .graphs.Amazon_Computers import Amazon_Computers
from .graphs.Amazon_Photo import Amazon_Photo
from .graphs.Coauthor_CS import Coauthor_CS
from .graphs.Coauthor_Physics import Coauthor_Physics
from .graphs.Reddit import Reddit
from .graphs.obgn_arxiv import obgn_arxiv
from .graphs.obgn_products import obgn_products
from .graphs.obgn_papers100M import obgn_papers100M



SEQ_DATASETS = {
    "Reuters10K": Reuters10K,
    "XYh5_scRNA": XYh5_scRNA,
    "ACM": ACM,
    "DBLP": DBLP,
    "Cora": Cora,
    "Citeseer": Citeseer,
    "Pubmed": Pubmed,
    "Wiki": Wiki,
    "BAT": BAT,
    "EAT": EAT,
    "UAT": UAT,
    "Amazon_Computers": Amazon_Computers,
    "Amazon_Photo": Amazon_Photo,
    "Coauthor_CS": Coauthor_CS,
    "Coauthor_Physics": Coauthor_Physics,
    "Reddit": Reddit,
    "obgn_arxiv": obgn_arxiv,
    "obgn_products": obgn_products,
    "obgn_papers100M": obgn_papers100M,
}
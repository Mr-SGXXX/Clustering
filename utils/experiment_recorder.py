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
from pyerm import Experiment

from .config import config

class ExperimentRecorder:
    def __init__(self, cfg:config, task_name:str="Clustering"):
        self.db_path = cfg['global']['pyerm_db_path']
        if self.db_path is None:
            return
        self.experiment = Experiment(self.db_path)
        self.experiment.task_init(task_name)
        dataset_name = cfg['global']['dataset']
        dataset_params = cfg[f'{dataset_name}']
        method_name = cfg['global']['method_name']
        method_params = cfg[f'{method_name}']
        global_params = cfg['global']
        global_params.pop('dataset')
        global_params.pop('method_name')
        method_params.update(global_params)
        self.experiment.data_init(dataset_name, dataset_params)
        self.experiment.method_init(method_name, method_params)
    
    def experiment_start(self, description, start_time):
        if self.db_path is None:
            return
        self.experiment.experiment_start(description, start_time)

    def experiment_over(self, rst_dict, images, end_time, useful_time_cost):
        if self.db_path is None:
            return
        self.experiment.experiment_over(rst_dict, images, end_time, useful_time_cost)
    
    def experiment_failed(self, exception:Exception, end_time=None):
        if self.db_path is None:
            return
        self.experiment.experiment_failed(str(exception), end_time)

    def detail_update(self, detail_dict):
        if self.db_path is None:
            return
        self.experiment.detail_update(detail_dict)

    


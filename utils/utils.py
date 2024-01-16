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
import os
import argparse
import torch
import numpy as np
import random
import logging
import typing

from .config import config
def get_args():
    """
    Get command line arguments.

    :return: argparse.Namespace, command line arguments
    """
    parser = argparse.ArgumentParser(description="Implementation of Clustering Methods, detailed setting is in the configuration file")
    parser.add_argument('-cp', '--config_path', default='./cfg/example.cfg', help="Configuration file path")
    parser.add_argument('-ss', '--split_symbol', default=',', help="Configuration file delimiter")
    args = parser.parse_args()
    cfg = config.init_by_path(args.config_path, split_symbol=args.split_symbol)
    return args, cfg

def seed_init(seed:typing.Union[None, int]=None):
    """
    Set random seed.
    :param seed: int, random seed, if None, do not set random seed
    """
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dir(cfg:config):
    """
    Create directories specified in the configuration.
    :param cfg: config object
    """
    method_name = cfg.get("global", "method_name")
    dataset_name = cfg.get("global", "dataset")
    log_dir = cfg.get("global", "log_dir")
    weight_dir = cfg.get("global", "weight_dir")
    result_dir = cfg.get("global", "result_dir")
    figure_dir = cfg.get("global", "figure_dir")
    log_dir = os.path.join(log_dir, method_name, dataset_name)
    cfg.set("global", "log_dir", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

def get_logger(log_dir:str, description:str, std_out:bool=True):
    """
    Create and return a logger object for recording and outputting log information.
    :param log_dir: str, log file storage path
    :param description: str, description of the run
    :param std_out: str, whether to output log information on the terminal
    
    :return: logging.Logger, logger object
    """
    # Configure log information format
    formatter = logging.Formatter('[%(asctime)s]: %(message)s')

    # Create logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # Set file handler
    log_dir = os.path.join(log_dir, f"{description}.log")
    handler = logging.FileHandler(log_dir)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # If std_out is True, set console handler
    if std_out:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger, log_dir


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
import torch
import numpy as np
import typing
import h5py
import os
import logging

from utils.metrics import Metrics
from .config import config


def save_rst(final_feature: typing.Union[None, np.ndarray, torch.Tensor],
             metrics: typing.Union[None, Metrics],
             y_pred: typing.Union[np.ndarray, torch.Tensor],
             y_true: typing.Union[None, np.ndarray, torch.Tensor],
             pretrain_time: float, train_time: float,
             description: str, logger:logging.Logger, cfg: config):
    method = cfg.get("global", "method_name")
    dataset = cfg.get("global", "dataset")
    result_path = os.path.join(
        cfg.get("global", "result_dir"), method, dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(result_path, f"{description}.h5")
    logger.info(f"Saving experiment result in {result_path}")
    if type(final_feature) is torch.Tensor:
        final_feature = final_feature.cpu().detach().numpy()
    if type(y_pred) is torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
    if type(y_true) is torch.Tensor:
        y_true = y_true.cpu().detach().numpy()
    with h5py.File(result_path, 'w') as fp:
        fp['y_pred'] = y_pred
        fp['y_true'] = y_true
        fp['pretrain_time'] = pretrain_time
        fp['train_time'] = train_time
        fp['final_feature'] = final_feature
        if metrics is not None:
            fp['acc'] = metrics.ACC.val_list
            fp['nmi'] = metrics.NMI.val_list
            fp['ari'] = metrics.ARI.val_list
            fp['sc'] = metrics.SC.val_list
            fp['comp'] = metrics.COMP.val_list
            fp['homo'] = metrics.HOMO.val_list
            for key in metrics.PretrainLoss:
                fp[f"pretrain_loss_{key}"] = metrics.PretrainLoss[key].val_list
            for key in metrics.Loss:
                fp[f"loss_{key}"] = metrics.Loss[key].val_list
    logger.info("Experiment result saved successfully!")

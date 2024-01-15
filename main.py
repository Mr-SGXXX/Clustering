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
import time
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import os
import traceback

from methods import *
from datasetLoader import DATASETS
from methods import CLASSICAL_METHODS, DEEP_METHODS, METHODS_INPUT_TYPES
from metrics import evaluate
from figures import draw_charts
from utils import email_reminder, get_logger, make_dir, get_args


def main():
    # initialize the configuration
    args, cfg = get_args()
    make_dir(cfg)
    start_time = time.time()
    description = (
        cfg.get("global", "description")
        if cfg.get("global", "description") is not None
        else (cfg.get("global", "method_name") + "_" +
        cfg.get("global", "dataset"))
    ) + f"_{int(start_time)}"
    print(f"Experiment: {description} is running...")
    logger, log_path = get_logger(
        cfg.get("global", "log_dir"), description, std_out=False)
    logger.info(
        f"Experiment {description}\tSeed {cfg.get('global', 'seed')}\tDevice {cfg.get('global', 'device')}"
    )
    reminder = email_reminder(
        cfg.get("global", "email_cfg_path"), logger=logger)
    try:
        # Select the method
        method = cfg.get("global", "method_name")
        if method in METHODS_INPUT_TYPES:
            method_input_types = METHODS_INPUT_TYPES[method]
        else:
            raise NotImplementedError(
                f"Method {method} not included in the `METHODS_INPUT_IMG_FLAG`")
        if method in CLASSICAL_METHODS:
            method = CLASSICAL_METHODS[method]
            method_flag = "classical"
        elif method in DEEP_METHODS:
            method = DEEP_METHODS[method]
            method_flag = "deep"
        else:
            raise NotImplementedError(
                f"Method {method} not included in the `CLASSICAL_METHODS` or `DEEP_METHODS`")

        # the Dataset Loading
        dataset = cfg.get("global", "dataset")
        if dataset in DATASETS:
            dataset = DATASETS[dataset](cfg, method_input_types)
        else:
            raise NotImplementedError(
                f"Dataset {dataset} not included in the `DATASETS`")
        logger.info("Dataset Init Over!")

        # the Clustering Training
        metrics = None
        features = None
        pretrain_features = None
        pretrain_start_time = None
        if method_flag == "classical":
            train_start_time = time.time()
            method = method(dataset, description, logger, cfg)
            pred_labels, features = method.fit()
            acc, nmi, ari, homo, comp = evaluate(pred_labels, dataset.label)
        elif method_flag == "deep":
            pretrain_start_time = time.time()
            method = method(dataset, description, logger, cfg)
            pretrain_features = method.pretrain()
            train_start_time = time.time()
            pred_labels, features = method.train_model()
            metrics = method.metrics
        else:
            raise NotImplementedError(f"Method Type {method_flag} Is Not Implemented!")

        train_end_time = time.time()

        # TODO: draw figures and save the results
        if metrics is not None:
            logger.info("Clustering Over!")
            logger.info(str(metrics))
            pretrain_start_time += metrics.pretrain_time_cost
            train_end_time -= metrics.clustering_time_cost
            logger.info(f"Pretrain Time Cost: {train_start_time - pretrain_start_time:.2f}s" if pretrain_start_time is not None else "")        
        else:
            logger.info(
                f"Clustering Over!\n" +
                f"Clustering Scores: ACC: {acc:.4f}\tNMI: {nmi:.4f}\tARI: {ari:.4f}\tHOMO: {homo:.4f}\tCOMP: {comp:.4f}" 
            )
        logger.info(f"Train Time Cost: {train_end_time - train_start_time:.2f}s")
        
        if cfg.get("global", "save_clustering_result") == True:
            rst_path = os.path.join(cfg.get("global", "result_dir"), f'{description}.csv')
            df = pd.DataFrame(pred_labels, columns=['Cluster'])
            df.to_csv(rst_path, index=True)
        try:
            figure_paths = draw_charts(
                rst_metrics=metrics,
                pretrain_features=pretrain_features,
                features=features,
                pred_labels=pred_labels,
                true_labels=dataset.label,
                description=description,
                cfg=cfg
            )
            logger.info(f"Figures Successfully Generated, saved in {figure_paths}!")
        except Exception as e:
            figure_paths = {}
            error_info = traceback.format_exc()
            logger.info(f"Figures Generation Failed for {error_info}")
        end_time = time.time()
        logger.info(f"Total Time Cost: {end_time - start_time:.2f}s")
        reminder.send_message(
            f"Experiment {description} is over.\n" +
            f"Method: {cfg.get('global', 'method_name')}\n" +
            f"Dataset: {cfg.get('global', 'dataset')}\n" +
            f"Metrics:\n{metrics}\n"
            f"Total time: {end_time - start_time:.2f}s\n" +
            (f"Pretrain time: {train_start_time - pretrain_start_time:.2f}s\n" if pretrain_start_time is not None else "") +
            f"Clustering time: {train_end_time - train_start_time:.2f}s\n" +
            "\n\n\n\n" + str(cfg),
            f'Experiment "{description}" Is Successfully Over',
            (log_path, *figure_paths),
        )
        print(f"Experiment: {description} is over...")
    except Exception as e:
        # raise e
        error_info = traceback.format_exc()
        logger.info(f"Experiment Going Wrong, Error: {e}\nFull traceback:{error_info}")
        reminder.send_message(
            f"Experiment {description} failed.\n" +
            f"Method: {cfg.get('global', 'method_name')}\n" +
            f"Dataset: {cfg.get('global', 'dataset')}\n" +
            f"Error Message: {e}\n" +
            f"Full traceback:{error_info}" +
            "\n\n\n\n" + str(cfg),
            f'Experiment "{description}" Failed for {e}',
            (log_path, ),
        )
        print(f"Experiment: {description} failed...")


if __name__ == "__main__":
    main()

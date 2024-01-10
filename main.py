import time
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import os
import traceback

from methods import *
from datasetProcesser import DATASETS
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
    print(description)
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
            method = method(cfg)
            train_start_time = time.time()
            pred_labels, features = method.fit(dataset.data)
            acc, nmi, ari, homo, comp = evaluate(pred_labels, dataset.label)
        elif method_flag == "deep":
            pretrain_start_time = time.time()
            method = method(dataset, description, logger, cfg)
            pretrain_features = method.pretrain()
            train_start_time = time.time()
            pred_labels, features = method.train_model()
            metrics = method.metrics

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
            logger.info("Figures Successfully Generated!")
        except Exception as e:
            figure_paths = None
            logger.info(f"Figures Generation Failed for {e}")
        end_time = time.time()
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


if __name__ == "__main__":
    main()

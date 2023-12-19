import time 

from methods import *
from methods import CLASSICAL_METHODS, DEEP_METHODS
from metrics import evaluate
from utils import config, email_reminder, get_logger, make_dir

def main():
    # initialize the configuration
    cfg = config.init_by_path("./cfg/test.cfg")
    make_dir(cfg)
    start_time = time.time()
    description = (
        cfg.get("global", "description")
        if cfg.get("global", "description") is not None
        else "experiment"
    ) + f"_{int(start_time)}"
    logger, log_path = get_logger(cfg.get("global", "log_dir"), description, std_out=False)
    logger.info(
        f"Experiment {description}\tSeed {cfg.get('global', 'seed')}\tDevice {cfg.get('global', 'device')}"
    )
    reminder = email_reminder(cfg.get("global", "email_cfg_path"), logger=logger)
    
    # sellect the method
    method = cfg.get("global", "method_name")
    if method in CLASSICAL_METHODS:
        method = CLASSICAL_METHODS[method]
        method_flag = "classical"
    elif method in DEEP_METHODS:
        method = DEEP_METHODS[method]
        method_flag = "deep"
    else:   
        raise NotImplementedError(f"Method {method} not implemented")
    
    # TODO: the Dataset Loading
    dataset = None
    
    # TODO: the Clustering Training

    if method_flag == "classical":
        method = method(cfg)
        pred_labels = method.fit(dataset.data)
        evaluate(pred_labels, dataset.label)
    elif method_flag == "deep":
        method = method(dataset, logger, cfg)
        method.pretrain()
        pred_labels, features, metrics = method.train()

    train_end_time = time.time()

    end_time = time.time()
    reminder.send_message(
        f"Experiment is over.\nTotal time: {end_time - start_time:.2f}s\tAverage time/epoch: {(train_end_time - start_time) / cfg.get('train', 'epoch'):.2f}s/epoch\n\n\n\n"
        + str(cfg),
        f'Experiment "{description}" is Successfully Over',
        (log_path),
    )


if __name__ == "__main__":
    main()


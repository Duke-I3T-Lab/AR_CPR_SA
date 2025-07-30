import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial

import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

from utils import utils

# from models.loss import l2_reg_loss
from data.dataset import (
    ImputationDataset,
    TransductionDataset,
    ClassiregressionDataset,
    collate_unsuperv,
    collate_superv,
)


logger = logging.getLogger("__main__")

NEG_METRICS = {"loss"}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config["task"]

    if task == "imputation":
        return (
            partial(
                ImputationDataset,
                mean_mask_length=config["mean_mask_length"],
                masking_ratio=config["masking_ratio"],
                mode=config["mask_mode"],
                distribution=config["mask_distribution"],
                exclude_feats=config["exclude_feats"],
            ),
            collate_unsuperv,
            UnsupervisedRunner,
        )
    if task == "transduction":
        return (
            partial(
                TransductionDataset,
                mask_feats=config["mask_feats"],
                start_hint=config["start_hint"],
                end_hint=config["end_hint"],
            ),
            collate_unsuperv,
            UnsupervisedRunner,
        )
    if (task == "classification") or (task == "regression"):
        return ClassiregressionDataset, collate_superv, None
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical(
                "Failed to load configuration file. Check JSON syntax and verify that files exist"
            )
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                output_dir
            )
        )

    output_dir = os.path.join(output_dir, config["experiment_name"])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["initial_timestamp"] = formatted_timestamp
    if (not config["no_timestamp"]) or (len(config["experiment_name"]) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config["output_dir"] = output_dir
    config["save_dir"] = os.path.join(output_dir, "checkpoints")
    config["pred_dir"] = os.path.join(output_dir, "predictions")
    config["tensorboard_dir"] = os.path.join(output_dir, "tb_summaries")
    utils.create_dirs(
        [config["save_dir"], config["pred_dir"], config["tensorboard_dir"]]
    )

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config

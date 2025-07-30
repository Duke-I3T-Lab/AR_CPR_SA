from transformers import PatchTSMixerConfig, PatchTSMixerForTimeSeriesClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import ClassiregressionDataset


# Project modules
from options import Options
from running import setup

# from utils import utils
from data.data import data_factory, Normalizer
from data.dataset import collate_superv
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(model, val_dataset, batch_size, device, num_classes=3):
    """
    Evaluate the model on the validation dataset and compute accuracy.

    Args:
        model: The trained model.
        val_dataset: The validation dataset.
        batch_size: Batch size for evaluation.
        device: The device (e.g., "cuda" or "cpu") to run evaluation on.

    Returns:
        accuracy: The accuracy of the model on the validation dataset.
    """
    # Set model to evaluation mode
    model.eval()

    # Create DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_huggingface,
    )

    correct_predictions = 0
    total_predictions = 0
    label_distribution = np.zeros(num_classes)
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in val_loader:
            # Move inputs and labels to the specified device
            inputs = batch["past_values"].to(device)
            labels = batch["target_values"]
            label_distribution += np.bincount(labels, minlength=num_classes)

            # Forward pass: Get logits from the model
            outputs = model(inputs)
            logits = outputs.prediction_outputs  # Extract logits from the output

            # Compute predictions by taking the argmax over logits
            predictions = torch.argmax(logits, dim=-1)
            # to cpu
            predictions = predictions.cpu().detach()

            # Update counts for accuracy calculation
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate and return accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy, label_distribution


# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)  # Get predicted class indices
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def collate_fn_huggingface(data, max_len=None):
    X, targets, padding_masks, IDs = collate_superv(data, max_len)
    targets = torch.argmax(targets, dim=1)
    return {"past_values": X, "target_values": targets}


def main(config):
    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)

    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run

    if config["seed"] is not None:
        torch.manual_seed(config["seed"])

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["gpu"] != "-1") else "cpu"
    )
    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]

    with open(f"my_data/split/user_split.json", "r") as f:
        print("config: ", config["split_id"])
        split_data = json.load(f)[f"split{config['split_id']}"]
        train_users = split_data["train_users"]
        test_users = split_data["test_users"]

        print(f"Loaded train users: {train_users}")
        print(f"Loaded test users: {test_users if test_users is not None else 'None'}")

    train_data = data_class(
        config["data_dir"], user_ids=train_users, n_proc=1, config=config, clean=True
    )
    val_data = data_class(
        config["data_dir"], user_ids=test_users, n_proc=-1, config=config
    )  # note that we use the extensive hyperparameter search, so it's essentially a val set.

    train_indices = train_data.all_IDs
    val_indices = val_data.all_IDs

    feat_dim = train_data.feature_df.shape[1]  # dimensionality of data features

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    with open(os.path.join(config["output_dir"], "data_indices.json"), "w") as f:
        try:
            json.dump(
                {
                    "train_indices": list(map(int, train_indices)),
                    "val_indices": list(map(int, val_indices)),
                },
                f,
                indent=4,
            )
        except ValueError:  # in case indices are non-integers
            json.dump(
                {
                    "train_indices": list(train_indices),
                    "val_indices": list(val_indices),
                },
                f,
                indent=4,
            )

    # Pre-process features
    normalizer = None
    if config["normalization"] is not None and config["normalization"] != "none":
        normalizer = Normalizer(config["normalization"])
        train_data.feature_df.loc[train_indices] = normalizer.normalize(
            train_data.feature_df.loc[train_indices]
        )

        norm_dict = normalizer.__dict__
        with open(
            os.path.join(config["output_dir"], "normalization.pickle"), "wb"
        ) as f:
            pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(
                val_data.feature_df.loc[val_indices]
            )

    logger.info("Creating model ...")
    model_config = PatchTSMixerConfig(
        context_length=420,  # window size
        patch_length=config[
            "patch_length"
        ],  # patch size, should evenly divide context_length
        num_input_channels=feat_dim,  # number of input features
        num_targets=config["num_classes"],  # label number
        patch_stride=config[
            "patch_length"
        ],  # patch stride, common to be equal to patch_length
        d_model=config["d_model"],
        num_layer=config["num_layers"],
        dropout=config["dropout"],
        head_aggregation="avg_pool",
    )
    model = PatchTSMixerForTimeSeriesClassification(model_config)

    logger.info("Model:\n{}".format(model))

    val_dataset = ClassiregressionDataset(
        val_data, val_indices
    )  # this is basically the test set

    train_dataset = ClassiregressionDataset(train_data, train_indices)

    training_args = TrainingArguments(
        output_dir=os.path.join(
            config["output_dir"], "metrics_" + config["experiment_name"]
        ),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(
            config["output_dir"], "logs_" + config["experiment_name"]
        ),
        learning_rate=config["lr"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["target_values"],
        report_to="tensorboard",
    )

    opt = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=config["lr"],
        steps_per_epoch=math.ceil(len(train_dataset) / config["batch_size"]),
        epochs=config["epochs"],
        pct_start=0.4,
    )  # this was found to work the best
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_huggingface,
        optimizers=(opt, scheduler),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    accuracy, label_distribution = evaluate_model(
        model,
        val_dataset,
        config["batch_size"],
        device,
        num_classes=config["num_classes"],
    )
    logger.info(f"Validation accuracy: {accuracy}")
    logger.info(f"Validation label distribution: {label_distribution}")

    return


if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    main(config)

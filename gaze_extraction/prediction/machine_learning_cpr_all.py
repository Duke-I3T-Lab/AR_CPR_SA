import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import warnings
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from utils import k_fold_parameter_search

warnings.filterwarnings("ignore")

# Constants
DURATION = 14
WINDOW_SIZE = 420
STEP_SIZE = 60
DATA_DIR = f"my_data/processed_ml_data_{DURATION}s"

USER_SPLIT_PATH = "my_data/split/user_split.json"


METRICS = {
    "fixation_metrics": ["MFD", "SFD", "FR", "PFT"],
    "saccade_metrics": ["MSV", "MPSV", "MSA", "MSD"],
    "blink_metrics": ["BR"],
    "roi_metrics": ["PFV", "VMFD", "VFR"],
}
ALL_FEATURE_COLUMNS = [
    metric_name
    for metric_type, metric_names in METRICS.items()
    for metric_name in metric_names
]
RANDOM_SEED = 42
NUM_SPLITS = 5
C = [0.1, 1, 10]
MAX_DEPTHS = [5, 7, 9, 11]
LEARNING_RATE = [0.1, 0.01]


def binarize_labels(y, threshold=0.5):
    return (y >= threshold).astype(int)


def scale(X_train, X_test, method="minmax"):
    scaling = get_scaler(method)

    X_train = scaling.fit_transform(X_train)
    X_test = scaling.transform(X_test)
    # convert back to dataframe
    X_train = pd.DataFrame(X_train, columns=ALL_FEATURE_COLUMNS)
    X_test = pd.DataFrame(X_test, columns=ALL_FEATURE_COLUMNS)
    return X_train, X_test


def get_scaler(method="minmax"):
    if method == "minmax":
        return MinMaxScaler()
    elif method == "maxabs":
        return MaxAbsScaler()
    elif method == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")


def load_user_split(split_file_path, split_id=4):
    with open(split_file_path, "r") as f:
        split_data = json.load(f)[f"split{split_id}"]
        train_users = split_data["train_users"]
        train_add_users = split_data.get("train_add_users", [])
        train_users += train_add_users
        test_users = (
            split_data["test_users"]
            if "test_users" in split_data and len(split_data["test_users"]) > 0
            else None
        )
        return train_users, test_users


def load_user_data(user_ids, event_types=None):
    """
    Load data for specified users and event types

    Args:
        user_ids: List of user IDs to load
        event_types: List of event types to load (e.g., ['vomiting', 'bleeding'])
                    If None, load all available events

    Returns:
        DataFrame with all data combined
    """
    all_data = []

    for user_id in user_ids:
        # Find all matching files for this user
        files = [
            f
            for f in os.listdir(DATA_DIR)
            if f.startswith(f"p{user_id}_") and f.endswith(".csv")
        ]

        # Filter by event type if specified
        if event_types:
            filtered_files = []
            for f in files:
                # Extract the event part from the filename (after the user_id prefix)
                event_part = f.split("_")[1].split(".")[0]
                if event_part in event_types:
                    filtered_files.append(f)
            files = filtered_files
        for file in files:
            try:
                # Load the CSV file
                file_path = os.path.join(DATA_DIR, file)
                df = pd.read_csv(file_path)

                # Add user ID and event type as columns
                event_type = file.split("_")[1].split(".")[0]
                df["user_id"] = user_id
                df["event"] = event_type

                all_data.append(df)
            except Exception as e:
                print(f"Error loading file {file}: {e}")

    if not all_data:
        raise ValueError("No data found for the specified users and events")

    return pd.concat(all_data, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="CPR Machine Learning - All Splits")
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=[
            "svm",
            "lr",
            "decision_tree",
            "random_forest",
            "adaboost_lr",
            "adaboost_tree",
        ],
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="none",
        choices=["minmax", "maxabs", "standard", "none"],
    )
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument(
        "--events",
        type=str,
        help="Comma-separated list of event types to include",
        default="vomiting, bleeding",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="gaze_extraction/logs/cpr_ml_all_splits_results.log",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of split IDs to run",
    )

    return parser.parse_args()


def setup_logger(log_file):
    logger = logging.getLogger("cpr_ml_all")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_and_evaluate_model(
    model_name, X_train, y_train, X_test, y_test, args, logger, split_id
):
    """Train and evaluate a model for a specific split, returning metrics focused on class 0"""

    # Apply oversampling if requested
    if args.oversampling and not model_name.lower().startswith("ada"):
        # logger.info("Applying SMOTE oversampling...")
        sm = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Apply scaling if requested
    if args.scaling != "none":
        # logger.info(f"Applying {args.scaling} scaling...")
        X_train, X_test = scale(X_train, X_test, method=args.scaling)

    # Train and evaluate based on the model
    model = model_name.lower()
    metrics = {}

    if model == "svm":
        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": [0.01, 0.1, 0.5, 1, 10],
            "kernel": ["rbf", "linear", "poly"],
        }
        grid = GridSearchCV(
            svm.SVC(random_state=RANDOM_SEED),
            param_grid,
            refit=True,
            verbose=0,
            cv=NUM_SPLITS,
        )
        grid.fit(X_train, y_train)
        # logger.info(f"Split {split_id}: SVM - best params: {grid.best_params_}")
        # logger.info(
        #     f"Split {split_id}: SVM - best validation score: {grid.best_score_:.4f}"
        # )

        # Evaluate on test set
        y_test_pred = grid.predict(X_test)

    elif model == "lr":
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.1, 0.5, 1],
            "solver": ["liblinear"],
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=RANDOM_SEED, multi_class="ovr"),
            param_grid,
            refit=True,
            verbose=0,
            cv=NUM_SPLITS,
        )
        grid.fit(X_train, y_train)
        # logger.info(
        #     f"Split {split_id}: Logistic Regression - best params: {grid.best_params_}"
        # )
        # logger.info(
        #     f"Split {split_id}: Logistic Regression - best validation score: {grid.best_score_:.4f}"
        # )

        # Evaluate on test set
        y_test_pred = grid.predict(X_test)

    elif model == "decision_tree":
        param_grid = {
            "max_depth": MAX_DEPTHS,
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            param_grid,
            refit=True,
            verbose=0,
            cv=NUM_SPLITS,
        )
        grid.fit(X_train, y_train)
        # logger.info(
        #     f"Split {split_id}: Decision Tree - best params: {grid.best_params_}"
        # )
        # logger.info(
        #     f"Split {split_id}: Decision Tree - best validation score: {grid.best_score_:.4f}"
        # )

        # Evaluate on test set
        y_test_pred = grid.predict(X_test)

    elif model == "random_forest":
        param_grid = {
            "max_depth": MAX_DEPTHS,
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED),
            param_grid,
            refit=True,
            verbose=0,
            cv=NUM_SPLITS,
        )
        grid.fit(X_train, y_train)
        # logger.info(
        #     f"Split {split_id}: Random Forest - best params: {grid.best_params_}"
        # )
        # logger.info(
        #     f"Split {split_id}: Random Forest - best validation score: {grid.best_score_:.4f}"
        # )

        # Evaluate on test set
        y_test_pred = grid.predict(X_test)

    elif model == "adaboost_lr":
        param_to_tune = [
            {
                "estimator": LogisticRegression(
                    penalty="l2",
                    C=c,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
                "learning_rate": lr,
                "n_estimators": 100,
            }
            for c in C
            for lr in LEARNING_RATE
        ]

        score_dct = k_fold_parameter_search(
            X_train.copy(),
            y_train.copy(),
            AdaBoostClassifier,
            param_to_tune,
            num_splits=NUM_SPLITS,
            use_oversampling=args.oversampling,
            scaling=get_scaler(args.scaling) if args.scaling != "none" else None,
            random_seed=RANDOM_SEED,
            return_dict=True,
            score_fn=accuracy_score,
        )

        chosen_parameters = max(score_dct, key=score_dct.get)
        # logger.info(
        #     f"Split {split_id}: AdaBoost LR - selected parameters: {chosen_parameters}"
        # )
        # logger.info(
        #     f"Split {split_id}: AdaBoost LR - best validation score: {score_dct[chosen_parameters]:.4f}"
        # )

        bdt = AdaBoostClassifier(
            estimator=chosen_parameters[0],
            n_estimators=100,
            learning_rate=chosen_parameters[1],
            random_state=RANDOM_SEED,
        )
        bdt.fit(X_train, y_train)

        y_test_pred = bdt.predict(X_test)

    elif model == "adaboost_tree":
        param_to_tune = [
            {
                "estimator": DecisionTreeClassifier(max_depth=i),
                "learning_rate": lr,
                "n_estimators": 100,
            }
            for i in MAX_DEPTHS
            for lr in LEARNING_RATE
        ]

        score_dct = k_fold_parameter_search(
            X_train.copy(),
            y_train.copy(),
            AdaBoostClassifier,
            param_to_tune,
            num_splits=NUM_SPLITS,
            use_oversampling=args.oversampling,
            scaling=get_scaler(args.scaling) if args.scaling != "none" else None,
            random_seed=RANDOM_SEED,
            return_dict=True,
            score_fn=accuracy_score,
        )

        chosen_parameters = max(score_dct, key=score_dct.get)
        # logger.info(
        #     f"Split {split_id}: AdaBoost Tree - selected parameters: {chosen_parameters}"
        # )
        # logger.info(
        #     f"Split {split_id}: AdaBoost Tree - best validation score: {score_dct[chosen_parameters]:.4f}"
        # )

        bdt = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=chosen_parameters[0].max_depth),
            n_estimators=100,
            learning_rate=chosen_parameters[1],
            random_state=RANDOM_SEED,
        )
        bdt.fit(X_train, y_train)

        y_test_pred = bdt.predict(X_test)

    else:
        logger.error(f"Unsupported model: {model}")
        return None

    # Calculate metrics with focus on class 0
    test_acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, labels=[0, 1]
    )

    # Store metrics (focus on class 0 as requested)
    metrics = {
        "accuracy": test_acc,
        "precision_class0": precision[0],
        "recall_class0": recall[0],
        "f1_class0": f1[0],
        "split_id": split_id,
    }

    # Log the results
    # logger.info(f"Split {split_id}: Test accuracy: {test_acc:.4f}")
    # logger.info(
    #     f"Split {split_id}: Class 0 - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}"
    # )
    # logger.info(
    #     f"Split {split_id}: Classification Report:\n{classification_report(y_test, y_test_pred)}"
    # )

    return metrics


def plot_metrics(all_metrics, model_name, args):
    """Create visualizations for the metrics across splits"""
    metrics_df = pd.DataFrame(all_metrics)

    # Create directory for plots if it doesn't exist
    plot_dir = "gaze_extraction/results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Set up the plot style
    plt.style.use("seaborn-v0_8")

    # Plot 1: Metrics across splits
    plt.figure(figsize=(12, 7))

    # Plot each metric
    for metric in ["accuracy", "precision_class0", "recall_class0", "f1_class0"]:
        plt.plot(
            metrics_df["split_id"],
            metrics_df[metric],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=metric,
        )

    # Add mean lines
    for metric in ["accuracy", "precision_class0", "recall_class0", "f1_class0"]:
        mean_value = metrics_df[metric].mean()
        plt.axhline(
            y=mean_value,
            color=plt.gca().lines[-1].get_color(),
            linestyle="--",
            alpha=0.7,
        )

    plt.title(f"Performance Metrics Across Splits - {model_name.upper()}", fontsize=15)
    plt.xlabel("Split ID", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(metrics_df["split_id"])
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Add scaling and oversampling info
    scaling_text = (
        f"Scaling: {args.scaling}" if args.scaling != "none" else "No scaling"
    )
    oversample_text = "With oversampling" if args.oversampling else "No oversampling"
    plt.figtext(0.02, 0.02, f"{scaling_text}, {oversample_text}", fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{model_name}_metrics_by_split.png", dpi=300)

    # Plot 2: Bar chart of average metrics with error bars
    plt.figure(figsize=(10, 6))

    # Prepare data
    metric_names = [
        "Accuracy",
        "Precision\n(Class 0)",
        "Recall\n(Class 0)",
        "F1-Score\n(Class 0)",
    ]
    metric_values = [
        metrics_df["accuracy"].mean(),
        metrics_df["precision_class0"].mean(),
        metrics_df["recall_class0"].mean(),
        metrics_df["f1_class0"].mean(),
    ]
    metric_errors = [
        metrics_df["accuracy"].std(),
        metrics_df["precision_class0"].std(),
        metrics_df["recall_class0"].std(),
        metrics_df["f1_class0"].std(),
    ]

    # Create bar chart
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = plt.bar(
        metric_names,
        metric_values,
        yerr=metric_errors,
        capsize=8,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title(f"Average Performance Metrics - {model_name.upper()}", fontsize=15)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    # Add scaling and oversampling info
    plt.figtext(0.02, 0.02, f"{scaling_text}, {oversample_text}", fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{model_name}_average_metrics.png", dpi=300)
    plt.close("all")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)

    # Parse split IDs
    split_ids = [int(split_id) for split_id in args.splits.split(",")]

    # Parse event types
    events = args.events.split(", ") if args.events else None

    # logger.info(f"Running for split IDs: {split_ids}")
    # logger.info(f"Model: {args.model}")
    # logger.info(f"Event types: {events}")
    # logger.info(f"Scaling: {args.scaling}")
    # logger.info(f"Oversampling: {args.oversampling}")

    # Initialize results storage
    all_metrics = []

    # Process each split
    for split_id in split_ids:
        # logger.info(f"======= Processing Split {split_id} =======")

        # Get user splits
        train_users, test_users = load_user_split(USER_SPLIT_PATH, split_id=split_id)

        # Load data
        train_data = load_user_data(train_users, events)
        test_data = load_user_data(test_users, events)

        # Prepare features and labels
        X_train = train_data[ALL_FEATURE_COLUMNS]
        y_train = binarize_labels(train_data["label"])
        X_test = test_data[ALL_FEATURE_COLUMNS]
        y_test = binarize_labels(test_data["label"])

        # Log class distribution
        train_dist = np.unique(y_train, return_counts=True)
        test_dist = np.unique(y_test, return_counts=True)
        logger.info(
            f"Split {split_id} - Train class distribution: {dict(zip(train_dist[0], train_dist[1]))}"
        )
        logger.info(
            f"Split {split_id} - Test class distribution: {dict(zip(test_dist[0], test_dist[1]))}"
        )

        # Train and evaluate model
        metrics = train_and_evaluate_model(
            args.model, X_train, y_train, X_test, y_test, args, logger, split_id
        )

        if metrics:
            all_metrics.append(metrics)

    # Calculate and log average metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate averages and standard deviations
        avg_accuracy = metrics_df["accuracy"].mean()
        std_accuracy = metrics_df["accuracy"].std()

        avg_precision = metrics_df["precision_class0"].mean()
        std_precision = metrics_df["precision_class0"].std()

        avg_recall = metrics_df["recall_class0"].mean()
        std_recall = metrics_df["recall_class0"].std()

        avg_f1 = metrics_df["f1_class0"].mean()
        std_f1 = metrics_df["f1_class0"].std()

        # Create a summary table
        summary_table = [
            [
                "Split",
                "Accuracy",
                "Precision (Class 0)",
                "Recall (Class 0)",
                "F1 (Class 0)",
            ]
        ]

        for idx, row in metrics_df.iterrows():
            summary_table.append(
                [
                    f"Split {int(row['split_id'])}",
                    f"{row['accuracy']:.4f}",
                    f"{row['precision_class0']:.4f}",
                    f"{row['recall_class0']:.4f}",
                    f"{row['f1_class0']:.4f}",
                ]
            )

        summary_table.append(
            [
                "Average",
                f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
                f"{avg_precision:.4f} ± {std_precision:.4f}",
                f"{avg_recall:.4f} ± {std_recall:.4f}",
                f"{avg_f1:.4f} ± {std_f1:.4f}",
            ]
        )

        # Print the summary table
        logger.info("\nSummary of Results:")
        logger.info(tabulate(summary_table, headers="firstrow", tablefmt="grid"))

    # logger.info("================= ALL SPLITS COMPLETED =================")


if __name__ == "__main__":
    main()

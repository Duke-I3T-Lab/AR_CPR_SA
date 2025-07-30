from dataset.dataset import FinalGazeDataset
from model.ENPool import ENPoolModel
from torch_geometric.loader import DataLoader
import argparse
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)


# Configure argument parser
parser = argparse.ArgumentParser(description="Train FixSAGraph model")
parser.add_argument("--dataset", type=str, default="cpr1", help="Dataset name")
parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden channels")
parser.add_argument("--in_channels", type=int, default=7, help="input channels")

parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate"
)  # need experiment
parser.add_argument(
    "--num_attn_heads", type=int, default=1, help="Number of attention heads"
)
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
parser.add_argument("--window_size", type=int, default=420, help="Window size")
parser.add_argument("--window_step", type=int, default=60, help="Window step")
parser.add_argument(
    "--spatial_threshold", type=float, default=0.065, help="Spatial threshold"
)

parser.add_argument("--split_id", type=int, default=1, help="Split ID to use")
parser.add_argument(
    "--save_dir",
    type=str,
    default="checkpoints",
    help="Directory to save model checkpoints",
)
parser.add_argument(
    "--log_dir", type=str, default="logs", help="Directory to save logs"
)
parser.add_argument("--seed", type=int, default=2025, help="Random seed")
parser.add_argument(
    "--edge_score_method",
    type=str,
    default="linear",
    choices=["lstm", "mlp", "linear"],
)

parser.add_argument(
    "--gnn",
    type=str,
    default="GATv2Conv",
    choices=["GATv2Conv", "GCNConv", "GINEConv", "TransformerConv"],
)
parser.add_argument("--edge_score_no_attr", action="store_true")
parser.add_argument("--bidirectional_temporal_edge", action="store_true")
parser.add_argument("--only_final_layer", action="store_true")
parser.add_argument(
    "--used_features",
    type=str,
    default="spherical_3d_centroid, eye_center, duration, target",
)
parser.add_argument("--lr_reduce", type=float, default=0.5)


# Set up logger
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    class Logger:
        def __init__(self, log_file):
            pass

        def log(self, message):
            print(message)

    return Logger(log_file)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    class_losses = []

    for data in tqdm(loader, desc="Training", unit="batch", leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        # Extract true labels and create prediction
        y = data.y.to(device)

        # Calculate classification loss
        class_loss = F.nll_loss(
            out,
            y,
        )
        class_losses.append(class_loss.item())

        # Combine losses with regularization
        loss = class_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * data.num_graphs
        pred = out.max(1)[1]
        total_correct += pred.eq(y).sum().item()
        total_samples += y.size(0)

        # Store predictions and labels for metrics
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    avg_class_loss = np.mean(class_losses)
    accuracy = total_correct / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    # return avg_loss, accuracy, precision, recall, f1
    return avg_class_loss, accuracy, precision, recall, f1


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    class_losses = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Validation", unit="batch", leave=False):
            data = data.to(device)

            # Forward pass
            out = model(data)

            # Extract true labels and create prediction
            y = data.y.to(device)
            # Calculate classification loss
            class_loss = F.nll_loss(
                out,
                y,
            )
            class_losses.append(class_loss.item())

            # Combine losses with regularization
            loss = class_loss

            # Track metrics
            total_loss += loss.item() * data.num_graphs
            pred = out.max(1)[1]

            total_correct += pred.eq(y).sum().item()
            total_samples += y.size(0)

            # Store predictions and labels for metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    avg_class_loss = np.mean(class_losses)
    accuracy = total_correct / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_class_loss, accuracy, precision, recall, f1, conf_matrix


def main():
    # Parse arguments
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = setup_logger(args.log_dir)
    logger.log(f"Using device: {device}")

    # Create directories
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set up TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configure = f"batch_size_{args.batch_size}_hidden_{args.hidden_channels}_layers_{args.num_layers}_dropout_{args.dropout}_lr_{args.lr}_norm_{args.norm}_score_{args.edge_score_method}"
    tb_dir = os.path.join(
        args.log_dir,
        f"{args.dataset}",
        f"{configure}",
        f"split_{args.split_id}",
    )
    writer = SummaryWriter(tb_dir)
    logger.log(f"TensorBoard logs will be saved to {tb_dir}")

    split_file = f"my_data/split/user_split.json"
    logger.log(f"Loading split from {split_file}")

    with open(split_file, "r") as f:
        split_data = json.load(f)
        split_key = f"split{args.split_id}"
        if split_key not in split_data:
            logger.log(f"Error: {split_key} not found in split data")
            return
        logger.log(f"Using split: {split_key}")
        train_users = split_data[split_key]["train_users"]
        test_users = split_data[split_key]["test_users"]

        logger.log(f"Train users: {train_users}")
        logger.log(f"Test users: {test_users}")

    # Log hyperparameters
    logger.log("Hyperparameters:")
    for arg in vars(args):
        logger.log(f"  {arg}: {getattr(args, arg)}")

    # Create datasets
    logger.log("Loading datasets...")

    train_dataset = FinalGazeDataset(
        root=f"data/{args.dataset}/processed_21s",
        user_ids=(train_users if args.train_val_together else train_users),
        window_size=args.window_size,
        window_step=args.window_step,
        spatial_threshold=args.spatial_threshold,
        directed_temporal_edges=not args.bidirectional_temporal_edge,
        used_features=args.used_features,
    )

    test_dataset = FinalGazeDataset(
        root=f"data/{args.dataset}/processed_21s",
        user_ids=test_users,
        window_size=args.window_size,
        window_step=args.window_step,
        spatial_threshold=args.spatial_threshold,
        directed_temporal_edges=not args.bidirectional_temporal_edge,
        used_features=args.used_features,
    )

    logger.log(f"Train dataset size: {len(train_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    attr_in_conv = args.gnn in ["GATv2Conv", "TransformerConv", "GINEConv"]
    model = ENPoolModel(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        edge_score_method=args.edge_score_method,
        dropout=args.dropout,
        gnn=args.gnn,
        use_edge_attr_in_conv=attr_in_conv,
        use_attr_in_edge_score=not args.edge_score_no_attr,
        edge_dim=3,
        num_heads=args.num_attn_heads,
        cat_layer_features=not args.only_final_layer,
        edge_attr_aggr="sum",
    ).to(device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create learning rate scheduler that halfs every 10 epochs
    if args.lr_reduce < 1:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=args.lr_reduce
        )

    # Training loop
    start_time = time.time()

    logger.log("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_prec, train_recall, train_f1 = train(
            model, train_loader, optimizer, device
        )

        test_loss, test_acc, test_prec, test_recall, test_f1, test_conf = validate(
            model, test_loader, device
        )

        # Update learning rate
        if args.lr_reduce < 1:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.log(f"Epoch: {epoch+1}/{args.epochs}")
        logger.log(
            f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        if test_loader:
            logger.log(
                f"  Test   - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}"
            )
        logger.log(f"  LR: {current_lr:.6f}")

        # Write to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        writer.add_scalar("Accuracy/train", train_acc, epoch)

        writer.add_scalar("F1/train", train_f1, epoch)

        writer.add_scalar("Precision/train", train_prec, epoch)

        writer.add_scalar("Recall/train", train_recall, epoch)

        writer.add_scalar("LearningRate", current_lr, epoch)

        if test_loader:
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("F1/test", test_f1, epoch)
            writer.add_scalar("Precision/test", test_prec, epoch)
            writer.add_scalar("Recall/test", test_recall, epoch)

    # Training summary
    training_time = time.time() - start_time
    logger.log(f"Training completed in {training_time:.2f}s")
    if test_loader:
        logger.log(f"Final test F1: {test_f1:.4f}")
        logger.log(f"Final test accuracy: {test_acc:.4f}")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()

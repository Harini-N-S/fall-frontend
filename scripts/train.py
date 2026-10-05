#!/usr/bin/env python3
"""
train.py - Train FallGRUClassifier on Temporal Window Features.

Key Principles:
    - Train split used strictly for gradient optimization.
    - Validation split used strictly for model selection (best Val F1) and LR scheduling.
    - Test split remains 100% UNTOUCHED (zero leakage).
    - Addresses class imbalance via pos_weight in BCEWithLogitsLoss computed ONLY from Train.
    - Tracks epoch-level loss, accuracy, precision, recall, and F1.
    - Implements Early Stopping (patience=7) and saves best checkpoint to models/best_fall_gru.pt.
    - Saves complete training history to results/ml_training_history.csv.
"""

import sys
import os
import csv
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from model import FallGRUClassifier, create_model, count_parameters
from dataset import FallTemporalDataset

FEATURES_DIR = BASE_DIR / "data" / "features"
SPLITS_JSON = BASE_DIR / "data" / "splits" / "splits.json"
FALL_EVENTS_CSV = BASE_DIR / "data" / "pseudo_labels_refined" / "fall_events.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
HISTORY_CSV = RESULTS_DIR / "ml_training_history.csv"
CHECKPOINT_PATH = MODELS_DIR / "best_fall_gru.pt"


def set_seed(seed: int = 42):
    """
    Ensures deterministic reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        features = batch["features"].to(device)  # [B, 32, 62]
        labels = batch["label"].to(device)        # [B]

        optimizer.zero_grad()
        logits = model(features)                  # [B]
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

        # Binary predictions using fixed 0.0 logit threshold (prob >= 0.5)
        preds = (logits >= 0.0).long()

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n_samples = max(1, len(all_targets))
    epoch_loss = total_loss / n_samples
    epoch_acc = float(accuracy_score(all_targets, all_preds))
    epoch_prec = float(precision_score(all_targets, all_preds, zero_division=0))
    epoch_rec = float(recall_score(all_targets, all_preds, zero_division=0))
    epoch_f1 = float(f1_score(all_targets, all_preds, zero_division=0))

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": epoch_prec,
        "recall": epoch_rec,
        "f1": epoch_f1,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates model on validation split (no gradient tracking).
    """
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        logits = model(features)
        loss = criterion(logits, labels.float())

        total_loss += loss.item() * len(labels)

        preds = (logits >= 0.0).long()

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n_samples = max(1, len(all_targets))
    epoch_loss = total_loss / n_samples
    epoch_acc = float(accuracy_score(all_targets, all_preds))
    epoch_prec = float(precision_score(all_targets, all_preds, zero_division=0))
    epoch_rec = float(recall_score(all_targets, all_preds, zero_division=0))
    epoch_f1 = float(f1_score(all_targets, all_preds, zero_division=0))

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": epoch_prec,
        "recall": epoch_rec,
        "f1": epoch_f1,
    }


def train_model(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    window_size: int = 32,
    stride: int = 8,
    fall_overlap_threshold: float = 0.25,
    patience: int = 7,
    seed: int = 42,
    input_size: int = 62,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> Dict[str, Any]:
    """
    Main training execution routine.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("                TEMPORAL FALL DETECTION MODEL TRAINING (STEP 6)")
    print("=" * 85)
    print(f"Device:                  {device}")
    print(f"Random Seed:             {seed}")
    print(f"Epochs:                  {epochs}")
    print(f"Batch Size:              {batch_size}")
    print(f"Learning Rate:           {lr}")
    print(f"Weight Decay:            {weight_decay}")
    print(f"Early Stopping Patience: {patience} epochs (monitoring Val F1)")
    print(f"Window Size / Stride:    {window_size} frames / {stride} frames")
    print(f"Fall Overlap Threshold:  {fall_overlap_threshold}")
    print("=" * 85)

    # 1. Dataset Loading (Train & Validation ONLY)
    print("\nLoading datasets...")
    train_dataset = FallTemporalDataset(
        split="train",
        window_size=window_size,
        stride=stride,
        fall_overlap_threshold=fall_overlap_threshold,
        features_dir=FEATURES_DIR,
        splits_path=SPLITS_JSON,
        fall_events_csv=FALL_EVENTS_CSV,
        preload=True,
    )

    val_dataset = FallTemporalDataset(
        split="validation",
        window_size=window_size,
        stride=stride,
        fall_overlap_threshold=fall_overlap_threshold,
        features_dir=FEATURES_DIR,
        splits_path=SPLITS_JSON,
        fall_events_csv=FALL_EVENTS_CSV,
        preload=True,
    )

    # Verify split boundaries
    train_recordings = set(train_dataset.recordings)
    val_recordings = set(val_dataset.recordings)
    overlap = train_recordings & val_recordings
    assert len(overlap) == 0, f"Critical error: Train and Validation sets overlap on {overlap}!"

    train_counts = train_dataset.get_label_counts()
    val_counts = val_dataset.get_label_counts()

    print(f"Train Dataset:      {len(train_dataset):,} windows across {len(train_recordings)} recordings")
    print(f"  - FALL (1):       {train_counts['FALL']:,} ({train_counts['FALL'] / max(1, len(train_dataset)) * 100:.2f}%)")
    print(f"  - NO_FALL (0):    {train_counts['NO_FALL']:,} ({train_counts['NO_FALL'] / max(1, len(train_dataset)) * 100:.2f}%)")

    print(f"Validation Dataset: {len(val_dataset):,} windows across {len(val_recordings)} recordings")
    print(f"  - FALL (1):       {val_counts['FALL']:,} ({val_counts['FALL'] / max(1, len(val_dataset)) * 100:.2f}%)")
    print(f"  - NO_FALL (0):    {val_counts['NO_FALL']:,} ({val_counts['NO_FALL'] / max(1, len(val_dataset)) * 100:.2f}%)")

    # 2. Compute Class Imbalance Weight (TRAIN SET ONLY)
    n_fall_train = train_counts["FALL"]
    n_no_fall_train = train_counts["NO_FALL"]
    assert n_fall_train > 0, "No positive fall windows found in training set!"

    pos_weight_value = float(n_no_fall_train / n_fall_train)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    print(f"\nClass Imbalance Calculation (TRAIN SET ONLY):")
    print(f"  pos_weight = {n_no_fall_train} / {n_fall_train} = {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 3. Create DataLoaders
    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Initialize Model
    model = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    total_params = count_parameters(model)
    print(f"Initialized FallGRUClassifier: {total_params:,} trainable parameters.")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler monitors validation loss to adjust learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # 6. Training Loop
    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    early_stopped = False

    print("\n" + "=" * 85)
    print(f"{'Epoch':<7} {'Train Loss':<12} {'Train F1':<11} {'Val Loss':<11} {'Val Prec':<11} {'Val Rec':<11} {'Val F1':<10} {'LR':<10}")
    print("-" * 85)

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train one epoch
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate one epoch
        val_res = evaluate_epoch(model, val_loader, criterion, device)

        # Step scheduler based on validation loss
        scheduler.step(val_res["loss"])

        # Record history row
        row = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "train_accuracy": round(train_res["accuracy"], 4),
            "train_precision": round(train_res["precision"], 4),
            "train_recall": round(train_res["recall"], 4),
            "train_f1": round(train_res["f1"], 4),
            "val_loss": round(val_res["loss"], 4),
            "val_accuracy": round(val_res["accuracy"], 4),
            "val_precision": round(val_res["precision"], 4),
            "val_recall": round(val_res["recall"], 4),
            "val_f1": round(val_res["f1"], 4),
            "learning_rate": current_lr,
        }
        history.append(row)

        print(
            f"{epoch:<7} "
            f"{train_res['loss']:<12.4f} "
            f"{train_res['f1']:<11.4f} "
            f"{val_res['loss']:<11.4f} "
            f"{val_res['precision']:<11.4f} "
            f"{val_res['recall']:<11.4f} "
            f"{val_res['f1']:<10.4f} "
            f"{current_lr:.1e}"
        )

        # Model selection: Best Validation F1
        if val_res["f1"] > best_val_f1 + 1e-5:
            best_val_f1 = val_res["f1"]
            best_epoch = epoch
            best_metrics = {
                "val_loss": val_res["loss"],
                "val_accuracy": val_res["accuracy"],
                "val_precision": val_res["precision"],
                "val_recall": val_res["recall"],
                "val_f1": val_res["f1"],
                "train_loss": train_res["loss"],
                "train_accuracy": train_res["accuracy"],
                "train_precision": train_res["precision"],
                "train_recall": train_res["recall"],
                "train_f1": train_res["f1"],
            }
            patience_counter = 0

            # Save best model checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_metrics": best_metrics,
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "window_size": window_size,
                    "stride": stride,
                    "fall_overlap_threshold": fall_overlap_threshold,
                    "patience": patience,
                },
                "random_seed": seed,
                "pos_weight": pos_weight_value,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping triggered: Validation F1 did not improve for {patience} consecutive epochs.")
                early_stopped = True
                break

    training_duration = time.time() - start_training_time

    # 7. Save History to CSV
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "learning_rate",
    ]
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"\nSaved training history to: {HISTORY_CSV}")
    print(f"Saved best checkpoint to:   {CHECKPOINT_PATH}")

    final_row = history[-1]

    # 8. Print Final Report
    print("\n" + "=" * 85)
    print("                       TRAINING RUN COMPLETE (REPORT)")
    print("=" * 85)
    print(f"Total Epochs Trained:            {len(history)}")
    print(f"Best Epoch:                      {best_epoch}")
    print(f"Best Validation F1:              {best_metrics.get('val_f1', 0.0):.4f}")
    print(f"Best Validation Precision:       {best_metrics.get('val_precision', 0.0):.4f}")
    print(f"Best Validation Recall:          {best_metrics.get('val_recall', 0.0):.4f}")
    print(f"Best Validation Accuracy:        {best_metrics.get('val_accuracy', 0.0):.4f}")
    print(f"Best Validation Loss:            {best_metrics.get('val_loss', 0.0):.4f}")
    print("-" * 85)
    print(f"Final Epoch ({len(history)}) Train Loss:        {final_row['train_loss']:.4f}")
    print(f"Final Epoch ({len(history)}) Train F1:          {final_row['train_f1']:.4f}")
    print(f"Final Epoch ({len(history)}) Val Loss:          {final_row['val_loss']:.4f}")
    print(f"Final Epoch ({len(history)}) Val F1:            {final_row['val_f1']:.4f}")
    print("-" * 85)
    print(f"Initial Learning Rate:           {lr:.1e}")
    print(f"Final Learning Rate:             {final_row['learning_rate']:.1e}")
    print(f"Class Weight (pos_weight):       {pos_weight_value:.4f}")
    print(f"Early Stopping Occurred:         {early_stopped}")
    print(f"Total Training Duration:         {training_duration:.2f}s ({training_duration / len(history):.2f}s / epoch)")
    print(f"Checkpoint File:                 {CHECKPOINT_PATH}")
    print("=" * 85)
    print("\nNOTE: Validation metrics represent validation performance against refined pseudo-labels,")
    print("not human-annotated ground truth.")
    print("=" * 85)

    return {
        "total_epochs": len(history),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_row,
        "pos_weight": pos_weight_value,
        "early_stopped": early_stopped,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }


def main():
    parser = argparse.ArgumentParser(description="Train FallGRUClassifier on temporal window features")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--window_size", type=int, default=32, help="Window length in frames (default: 32)")
    parser.add_argument("--stride", type=int, default=8, help="Window stride in frames (default: 8)")
    parser.add_argument("--fall_overlap_threshold", type=float, default=0.25, help="Overlap ratio for fall label (default: 0.25)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        stride=args.stride,
        fall_overlap_threshold=args.fall_overlap_threshold,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

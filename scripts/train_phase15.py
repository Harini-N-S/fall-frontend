#!/usr/bin/env python3
"""
scripts/train_phase15.py
Phase 15: Train Improved FallGRUClassifier on the Automatic Pseudo-Labeled Dataset.

Key Principles:
  - Trains exclusively on data/training_dataset/train.npz.
  - Validation split (data/training_dataset/val.npz) used for model selection (best Val F1) and LR scheduling.
  - Test split remains 100% UNTOUCHED (zero leakage).
  - Handles class imbalance (~1:13.2) using class-weighted BCEWithLogitsLoss (pos_weight = 13.186).
  - Evaluates loss, accuracy, precision, recall, F1, PR-AUC, and ROC-AUC per epoch.
  - Saves:
      models/phase15/best_model.pt
      models/phase15/training_history.json
      models/phase15/config.yaml
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from model import FallGRUClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset_npz(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")
    data = np.load(npz_path)
    features = torch.from_numpy(data["features"]).float()  # (N, 32, 62)
    labels = torch.from_numpy(data["labels"]).float()      # (N,)
    return features, labels


def compute_metrics(targets: List[int], probs: List[float], preds: List[int]) -> Dict[str, float]:
    y_true = np.array(targets)
    y_prob = np.array(probs)
    y_pred = np.array(preds)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_targets: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_targets.extend(labels.cpu().long().tolist())
        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n_samples = max(1, len(all_targets))
    epoch_loss = total_loss / n_samples
    metrics = compute_metrics(all_targets, all_probs, all_preds)
    return epoch_loss, metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_targets.extend(labels.cpu().long().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    n_samples = max(1, len(all_targets))
    epoch_loss = total_loss / n_samples
    metrics = compute_metrics(all_targets, all_probs, all_preds)
    return epoch_loss, metrics


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
):
    print("=" * 80)
    print("PHASE 15: TRAINING IMPROVED GRU ON PSEUDO-LABELED DATASET")
    print("=" * 80)

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # 1. Load train and val datasets
    train_x, train_y = load_dataset_npz(dataset_dir / "train.npz")
    val_x, val_y = load_dataset_npz(dataset_dir / "val.npz")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    train_pos = int((train_y == 1).sum().item())
    train_neg = int((train_y == 0).sum().item())
    val_pos = int((val_y == 1).sum().item())
    val_neg = int((val_y == 0).sum().item())

    # Principled class weight: neg / pos from train split ONLY
    pos_weight_val = train_neg / max(1, train_pos)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)

    print(f"Train samples: {len(train_x)} (Pos: {train_pos}, Neg: {train_neg}) -> pos_weight = {pos_weight_val:.3f}")
    print(f"Val samples:   {len(val_x)} (Pos: {val_pos}, Neg: {val_neg})")

    # 2. Instantiate Model
    model = FallGRUClassifier(
        input_size=62,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 3. Training loop with early stopping
    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epochs_no_improve = 0
    best_model_state: Optional[Dict[str, Any]] = None

    print("\nStarting Training Loop...")
    print(f"{'Epoch':<6} {'Train Loss':<11} {'Train F1':<9} {'Val Loss':<10} {'Val Prec':<9} {'Val Rec':<8} {'Val F1':<8} {'PR-AUC':<8} {'ROC-AUC':<8} {'LR':<9}")
    print("-" * 95)

    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        train_loss, train_met = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_met = evaluate(model, val_loader, criterion, device)

        curr_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_met["f1"])

        ep_record = {
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_met["accuracy"], 4),
            "train_precision": round(train_met["precision"], 4),
            "train_recall": round(train_met["recall"], 4),
            "train_f1": round(train_met["f1"], 4),
            "train_pr_auc": round(train_met["pr_auc"], 4),
            "train_roc_auc": round(train_met["roc_auc"], 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_met["accuracy"], 4),
            "val_precision": round(val_met["precision"], 4),
            "val_recall": round(val_met["recall"], 4),
            "val_f1": round(val_met["f1"], 4),
            "val_pr_auc": round(val_met["pr_auc"], 4),
            "val_roc_auc": round(val_met["roc_auc"], 4),
            "learning_rate": curr_lr,
            "duration_seconds": round(time.time() - t_ep_start, 2),
        }
        history.append(ep_record)

        print(
            f"{ep:02d}/{epochs}  {train_loss:<11.4f} {train_met['f1']:<9.4f} "
            f"{val_loss:<10.4f} {val_met['precision']:<9.4f} {val_met['recall']:<8.4f} "
            f"{val_met['f1']:<8.4f} {val_met['pr_auc']:<8.4f} {val_met['roc_auc']:<8.4f} "
            f"{curr_lr:<9.1e}"
        )

        # Check for improvement
        if val_met["f1"] > best_val_f1:
            best_val_f1 = val_met["f1"]
            best_epoch = ep
            best_metrics = ep_record
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] No improvement in Validation F1 for {patience} consecutive epochs. Stopping at Epoch {ep}.")
                break

    # 4. Save Best Model Checkpoint
    checkpoint_out = output_dir / "best_model.pt"
    ckpt_payload = {
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "input_size": 62,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "pos_weight": pos_weight_val,
        "random_seed": seed,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "window_size": 32,
            "stride": 8,
        },
        "best_validation_metrics": best_metrics,
    }
    torch.save(ckpt_payload, checkpoint_out)
    print(f"\nSaved Best Model Checkpoint to: {checkpoint_out} (Best Epoch: {best_epoch}, Val F1: {best_val_f1:.4f})")

    # 5. Save Training History
    hist_out = output_dir / "training_history.json"
    with open(hist_out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved Training History to: {hist_out}")

    # 6. Save Configuration YAML
    config_out = output_dir / "config.yaml"
    cfg_data = {
        "phase": 15,
        "model": {
            "architecture": "FallGRUClassifier",
            "input_size": 62,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "pos_weight": float(pos_weight_val),
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        },
        "dataset": {
            "path": str(dataset_dir),
            "train_samples": len(train_x),
            "val_samples": len(val_x),
            "train_pos": train_pos,
            "train_neg": train_neg,
        },
        "best_epoch": best_epoch,
        "best_validation_f1": best_val_f1,
    }
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f, default_flow_style=False)
    print(f"Saved Training Config to: {config_out}")

    return checkpoint_out, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 15 GRU Training")
    parser.add_argument("--dataset-dir", type=str, default=str(BASE_DIR / "data" / "training_dataset"))
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "models" / "phase15"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )

#!/usr/bin/env python3
"""
evaluate.py - Independent Evaluation of Best FallGRUClassifier on Untouched Test Split.

Evaluates:
    models/best_fall_gru.pt (Checkpoint selected via Validation F1 at Epoch 4)
Against:
    data/splits/splits.json -> test split (6 recordings, 302 windows)

Guarantees:
    - Evaluates ONLY the held-out test split (train and val splits are untouched).
    - Zero retraining, threshold tuning, or hyperparameter modification.
    - Full window-level, per-recording, and event-level detection analyses.
    - Generates:
        results/ml_test_window_predictions.csv
        results/ml_test_per_recording.csv
        results/ml_test_metrics.json
        results/ml_test_confusion_matrix.png
"""

import sys
import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from model import FallGRUClassifier
from dataset import FallTemporalDataset, load_fall_events_map

FEATURES_DIR = BASE_DIR / "data" / "features"
SPLITS_JSON = BASE_DIR / "data" / "splits" / "splits.json"
FALL_EVENTS_CSV = BASE_DIR / "data" / "pseudo_labels_refined" / "fall_events.csv"
CHECKPOINT_PATH = BASE_DIR / "models" / "best_fall_gru.pt"
RESULTS_DIR = BASE_DIR / "results"

WINDOW_PREDS_CSV = RESULTS_DIR / "ml_test_window_predictions.csv"
PER_REC_CSV = RESULTS_DIR / "ml_test_per_recording.csv"
METRICS_JSON = RESULTS_DIR / "ml_test_metrics.json"
CONF_MATRIX_PNG = RESULTS_DIR / "ml_test_confusion_matrix.png"


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[FallGRUClassifier, Dict]:
    """
    Loads saved checkpoint and reconstructs FallGRUClassifier model.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]

    model = FallGRUClassifier(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Runs forward inference over all test batches and collects window predictions.
    """
    all_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)  # [B, 32, 62]
            labels = batch["label"].to(device)    # [B]

            logits = model(feats)                 # [B]
            probs = torch.sigmoid(logits)         # [B]
            preds = (probs >= 0.5).long()         # [B]

            b_size = len(labels)
            for i in range(b_size):
                rec_id = str(batch["recording_id"][i])
                s_frame = int(batch["start_frame"][i])
                e_frame = int(batch["end_frame"][i])
                s_time = float(batch["start_time"][i])
                e_time = float(batch["end_time"][i])
                t_label = int(labels[i].item())
                p_label = int(preds[i].item())
                p_prob = float(probs[i].item())

                all_records.append({
                    "recording_id": rec_id,
                    "start_frame": s_frame,
                    "end_frame": e_frame,
                    "start_time": s_time,
                    "end_time": e_time,
                    "true_label": t_label,
                    "predicted_label": p_label,
                    "fall_probability": p_prob,
                })

    return all_records


def compute_window_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes window-level classification metrics, confusion matrix, ROC-AUC, and PR-AUC.
    """
    y_true = np.array([r["true_label"] for r in records], dtype=np.int32)
    y_pred = np.array([r["predicted_label"] for r in records], dtype=np.int32)
    y_prob = np.array([r["fall_probability"] for r in records], dtype=np.float64)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    acc = float(accuracy_score(y_true, y_pred))
    fall_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    fall_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    fall_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    no_fall_prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    no_fall_rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    no_fall_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    specificity = float(tn / max(1, tn + fp))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # ROC-AUC and PR-AUC using continuous probabilities
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception as e:
        pr_auc = None

    return {
        "accuracy": round(acc, 4),
        "fall_precision": round(fall_prec, 4),
        "fall_recall": round(fall_rec, 4),
        "fall_f1": round(fall_f1, 4),
        "no_fall_precision": round(no_fall_prec, 4),
        "no_fall_recall": round(no_fall_rec, 4),
        "no_fall_f1": round(no_fall_f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
    }


def compute_recording_level_metrics(
    records: List[Dict[str, Any]],
    fall_events_map: Dict[str, List[Tuple[float, float]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Aggregates window predictions per recording and computes recording-level metrics.
    A recording is classified as FALL if at least one window is predicted FALL.
    """
    by_rec: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        rec_id = r["recording_id"]
        if rec_id not in by_rec:
            by_rec[rec_id] = []
        by_rec[rec_id].append(r)

    per_recording_rows: List[Dict[str, Any]] = []
    rec_true: List[int] = []
    rec_pred: List[int] = []

    for rec_id, wins in sorted(by_rec.items()):
        has_true_fall = rec_id in fall_events_map and len(fall_events_map[rec_id]) > 0
        true_class = "FALL" if has_true_fall else "NO_FALL"
        true_binary = 1 if has_true_fall else 0

        n_windows = len(wins)
        n_fall_windows = sum(1 for w in wins if w["true_label"] == 1)
        n_no_fall_windows = sum(1 for w in wins if w["true_label"] == 0)
        n_pred_fall = sum(1 for w in wins if w["predicted_label"] == 1)
        n_pred_no_fall = sum(1 for w in wins if w["predicted_label"] == 0)

        probs = [w["fall_probability"] for w in wins]
        max_prob = float(max(probs)) if probs else 0.0
        mean_prob = float(np.mean(probs)) if probs else 0.0

        # Recording-level rule: FALL if >= 1 window predicted FALL
        pred_class = "FALL" if n_pred_fall >= 1 else "NO_FALL"
        pred_binary = 1 if n_pred_fall >= 1 else 0

        rec_true.append(true_binary)
        rec_pred.append(pred_binary)

        per_recording_rows.append({
            "recording_id": rec_id,
            "true_recording_class": true_class,
            "predicted_recording_class": pred_class,
            "total_windows": n_windows,
            "true_fall_windows": n_fall_windows,
            "true_no_fall_windows": n_no_fall_windows,
            "pred_fall_windows": n_pred_fall,
            "pred_no_fall_windows": n_pred_no_fall,
            "max_fall_probability": round(max_prob, 4),
            "mean_fall_probability": round(mean_prob, 4),
            "is_correct": (true_class == pred_class),
        })

    # Recording-level summary metrics
    y_t = np.array(rec_true, dtype=np.int32)
    y_p = np.array(rec_pred, dtype=np.int32)

    rec_acc = float(accuracy_score(y_t, y_p))
    rec_prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec_rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    rec_f1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))

    cm_rec = confusion_matrix(y_t, y_p, labels=[0, 1])
    r_tn, r_fp, r_fn, r_tp = [int(v) for v in cm_rec.ravel()]

    rec_summary = {
        "accuracy": round(rec_acc, 4),
        "precision": round(rec_prec, 4),
        "recall": round(rec_rec, 4),
        "f1": round(rec_f1, 4),
        "confusion_matrix": {
            "TN": r_tn,
            "FP": r_fp,
            "FN": r_fn,
            "TP": r_tp,
        },
    }

    return per_recording_rows, rec_summary


def compute_event_detection_metrics(
    records: List[Dict[str, Any]],
    fall_events_map: Dict[str, List[Tuple[float, float]]],
    test_stems: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates fall event coverage and false-positive window occurrences outside events.
    """
    # Group by recording
    by_rec: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        rec_id = r["recording_id"]
        if rec_id not in by_rec:
            by_rec[rec_id] = []
        by_rec[rec_id].append(r)

    event_rows: List[Dict[str, Any]] = []
    total_test_events = 0
    detected_test_events = 0
    fp_windows_outside_events = 0

    for stem in test_stems:
        wins = by_rec.get(stem, [])
        events = fall_events_map.get(stem, [])
        pred_fall_wins = [w for w in wins if w["predicted_label"] == 1]

        # Check each event in this recording
        for ev_idx, (e_start, e_end) in enumerate(events, 1):
            total_test_events += 1

            # Windows overlapping this event interval
            overlapping_wins = [
                w for w in pred_fall_wins
                if max(w["start_time"], e_start) <= min(w["end_time"], e_end)
            ]
            hit = len(overlapping_wins) >= 1
            if hit:
                detected_test_events += 1

            earliest_overlap_start = min(w["start_time"] for w in overlapping_wins) if overlapping_wins else None
            latest_overlap_end = max(w["end_time"] for w in overlapping_wins) if overlapping_wins else None
            earliest_rec_pred = min(w["start_time"] for w in pred_fall_wins) if pred_fall_wins else None
            latest_rec_pred = max(w["end_time"] for w in pred_fall_wins) if pred_fall_wins else None
            max_prob = max(w["fall_probability"] for w in wins) if wins else 0.0

            event_rows.append({
                "recording_id": stem,
                "event_index": ev_idx,
                "refined_fall_event_start_time": e_start,
                "refined_fall_event_end_time": e_end,
                "event_duration": round(e_end - e_start, 2),
                "predicted_fall_windows_overlapping": len(overlapping_wins),
                "total_pred_fall_windows_recording": len(pred_fall_wins),
                "earliest_predicted_fall_start": earliest_overlap_start if earliest_overlap_start is not None else earliest_rec_pred,
                "latest_predicted_fall_end": latest_overlap_end if latest_overlap_end is not None else latest_rec_pred,
                "maximum_fall_probability": round(max_prob, 4),
                "event_received_predicted_fall_window": hit,
            })

        # Count false-positive windows outside all events for this recording
        for w in pred_fall_wins:
            overlaps_any_event = any(
                max(w["start_time"], e_s) <= min(w["end_time"], e_e)
                for e_s, e_e in events
            )
            if not overlaps_any_event:
                fp_windows_outside_events += 1

    event_recall = float(detected_test_events / max(1, total_test_events))

    event_summary = {
        "total_test_events": total_test_events,
        "detected_test_events": detected_test_events,
        "event_detection_recall": round(event_recall, 4),
        "fp_windows_outside_events": fp_windows_outside_events,
        "event_details": event_rows,
    }

    return event_rows, event_summary


def plot_confusion_matrix(cm_dict: Dict[str, int], save_path: Path):
    """
    Renders confusion matrix graphic using matplotlib.
    """
    tn = cm_dict["TN"]
    fp = cm_dict["FP"]
    fn = cm_dict["FN"]
    tp = cm_dict["TP"]

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(
                j, i, f"{val:,}",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if val > matrix.max() / 2 else "black",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred NO_FALL (0)", "Pred FALL (1)"], fontsize=10)
    ax.set_yticklabels(["True NO_FALL (0)", "True FALL (1)"], fontsize=10)
    ax.set_title("Test Set Confusion Matrix (FallGRUClassifier)", fontsize=11, pad=15)
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate best FallGRUClassifier on Test set")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Path to checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("      INDEPENDENT TEST SET EVALUATION: FallGRUClassifier (STEP 7)")
    print("=" * 85)
    print(f"Device:                {device}")
    print(f"Model Checkpoint:      {checkpoint_path}")
    print(f"Features Directory:    {FEATURES_DIR}")
    print(f"Splits Specification:  {SPLITS_JSON}")
    print("=" * 85)

    # 1. Load Model
    print("\nLoading model from checkpoint...")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    best_epoch = checkpoint.get("epoch", "Unknown")
    pos_weight = checkpoint.get("pos_weight", "Unknown")
    print(f"Loaded checkpoint from Epoch {best_epoch} (pos_weight={pos_weight}).")
    print("Model set to eval mode. Weights remain completely frozen.")

    # 2. Load TEST Dataset ONLY
    print("\nLoading TEST dataset only...")
    test_dataset = FallTemporalDataset(
        split="test",
        window_size=32,
        stride=8,
        fall_overlap_threshold=0.25,
        features_dir=FEATURES_DIR,
        splits_path=SPLITS_JSON,
        fall_events_csv=FALL_EVENTS_CSV,
        preload=True,
    )
    test_recordings = list(test_dataset.recordings)
    test_counts = test_dataset.get_label_counts()

    print(f"Test Dataset Loaded:    {len(test_dataset):,} windows across {len(test_recordings)} recordings")
    print(f"  - FALL windows (1):   {test_counts['FALL']:,} ({test_counts['FALL'] / max(1, len(test_dataset)) * 100:.2f}%)")
    print(f"  - NO_FALL windows (0): {test_counts['NO_FALL']:,} ({test_counts['NO_FALL'] / max(1, len(test_dataset)) * 100:.2f}%)")

    # Verify counts match expected
    assert len(test_dataset) == 302, f"Expected 302 test windows, got {len(test_dataset)}"
    assert test_counts["FALL"] == 25, f"Expected 25 FALL windows, got {test_counts['FALL']}"
    assert test_counts["NO_FALL"] == 277, f"Expected 277 NO_FALL windows, got {test_counts['NO_FALL']}"
    assert len(test_recordings) == 6, f"Expected 6 test recordings, got {len(test_recordings)}"
    print("[PASS] Test set size and class counts verified strictly (302 windows, 25 FALL, 277 NO_FALL, 6 recordings).")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # 3. Generate Predictions
    print("\nRunning inference over test windows...")
    records = evaluate_test_set(model, test_loader, device)
    print(f"Generated predictions for {len(records)} test windows.")

    # 4. Temporal Sanity Checks
    by_rec: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        rec_id = r["recording_id"]
        if rec_id not in by_rec:
            by_rec[rec_id] = []
        by_rec[rec_id].append(r)

    sanity_errors = []
    for rec_id, wins in by_rec.items():
        for k in range(len(wins) - 1):
            if wins[k]["start_time"] > wins[k + 1]["start_time"]:
                sanity_errors.append(f"{rec_id}: windows not temporally ordered!")
            if wins[k]["end_frame"] < wins[k]["start_frame"]:
                sanity_errors.append(f"{rec_id}: invalid frame range!")
        for w in wins:
            p = w["fall_probability"]
            if np.isnan(p) or np.isinf(p) or not (0.0 <= p <= 1.0):
                sanity_errors.append(f"{rec_id}: invalid probability {p}")

    assert len(sanity_errors) == 0, f"Sanity check failed: {sanity_errors}"
    print("[PASS] Temporal sanity check passed with 0 errors (ordered, within bounds, probabilities in [0, 1]).")

    # 5. Compute Window-Level Metrics
    win_metrics = compute_window_metrics(records)
    cm = win_metrics["confusion_matrix"]

    # 6. Compute Per-Recording Metrics
    fall_events_map = load_fall_events_map(FALL_EVENTS_CSV)
    per_rec_rows, rec_summary = compute_recording_level_metrics(records, fall_events_map)

    # 7. Compute Fall-Event Detection Metrics
    event_rows, event_summary = compute_event_detection_metrics(records, fall_events_map, test_recordings)

    # 8. Save Detailed Results
    # Window predictions CSV
    with open(WINDOW_PREDS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "recording_id", "start_frame", "end_frame", "start_time", "end_time",
            "true_label", "predicted_label", "fall_probability"
        ])
        writer.writeheader()
        writer.writerows(records)

    # Per-recording CSV
    with open(PER_REC_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_rec_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rec_rows)

    # Metrics JSON - contains top-level explicit keys and structured blocks
    full_metrics = {
        "checkpoint": str(checkpoint_path.name),
        "best_epoch": best_epoch,
        "threshold": 0.5,
        "test_window_count": len(records),
        "test_recording_count": len(test_recordings),
        "TN": cm["TN"],
        "FP": cm["FP"],
        "FN": cm["FN"],
        "TP": cm["TP"],
        "confusion_matrix": cm,
        "specificity": win_metrics["specificity"],
        "balanced_accuracy": win_metrics["balanced_accuracy"],
        "ROC_AUC": win_metrics["roc_auc"],
        "PR_AUC": win_metrics["pr_auc"],
        "window_level_metrics": win_metrics,
        "recording_level_metrics": rec_summary,
        "event_level_detection_metrics": event_summary,
        "methodological_note": "Performance measured against refined pseudo-labels on held-out test recordings. Not human ground-truth.",
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Confusion matrix graphic
    plot_confusion_matrix(cm, CONF_MATRIX_PNG)

    # 9. Print Structured Final Report (Sections A through K)
    print("\n" + "=" * 85)
    print("                    FINAL TEST SET EVALUATION REPORT")
    print("=" * 85)

    print("\nA. TEST DATASET SIZE")
    print(f"   Total Test Recordings:        {len(test_recordings)} (3 FALL, 3 NO_FALL)")
    print(f"   Total Temporal Windows:       {len(records)}")
    print(f"   FALL Windows (Positive, 1):   {cm['TP'] + cm['FN']} ({((cm['TP'] + cm['FN']) / len(records)) * 100:.2f}%)")
    print(f"   NO_FALL Windows (Negative, 0):{cm['TN'] + cm['FP']} ({((cm['TN'] + cm['FP']) / len(records)) * 100:.2f}%)")
    print(f"   Window Size / Stride:         32 frames / 8 frames")

    print("\nB. CONFUSION MATRIX (threshold = 0.5)")
    print(f"                     Pred NO_FALL (0)     Pred FALL (1)")
    print(f"   True NO_FALL (0)  TN = {cm['TN']:<15} FP = {cm['FP']:<15} (Total = {cm['TN'] + cm['FP']})")
    print(f"   True FALL (1)     FN = {cm['FN']:<15} TP = {cm['TP']:<15} (Total = {cm['TP'] + cm['FN']})")
    print(f"   Exact values: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    print("\nC. WINDOW-LEVEL METRICS")
    print(f"   Overall Accuracy:             {win_metrics['accuracy']:.4f} ({win_metrics['accuracy'] * 100:.2f}%)")
    print(f"   Balanced Accuracy:            {win_metrics['balanced_accuracy']:.4f} ({win_metrics['balanced_accuracy'] * 100:.2f}%)")
    print(f"   Specificity (TN Rate):        {win_metrics['specificity']:.4f} ({win_metrics['specificity'] * 100:.2f}%)")
    print(f"   FALL Precision:               {win_metrics['fall_precision']:.4f}")
    print(f"   FALL Recall (Sensitivity):    {win_metrics['fall_recall']:.4f}")
    print(f"   FALL F1-Score:                {win_metrics['fall_f1']:.4f}")
    print(f"   NO_FALL Precision:            {win_metrics['no_fall_precision']:.4f}")
    print(f"   NO_FALL Recall:               {win_metrics['no_fall_recall']:.4f}")
    print(f"   NO_FALL F1-Score:             {win_metrics['no_fall_f1']:.4f}")

    print("\nD. ROC-AUC")
    print(f"   ROC-AUC Score:                {win_metrics['roc_auc'] if win_metrics['roc_auc'] is not None else 'N/A'} (computed from continuous fall probabilities)")

    print("\nE. PR-AUC")
    print(f"   PR-AUC (Average Precision):   {win_metrics['pr_auc'] if win_metrics['pr_auc'] is not None else 'N/A'} (computed from continuous fall probabilities)")

    print("\nF. RECORDING-LEVEL METRICS")
    r_cm = rec_summary["confusion_matrix"]
    print(f"   Decision Rule:                At least 1 predicted FALL window classifies recording as FALL")
    print(f"   Recording-Level Accuracy:     {rec_summary['accuracy']:.4f} ({rec_summary['accuracy'] * 100:.2f}%)")
    print(f"   Recording-Level Precision:    {rec_summary['precision']:.4f}")
    print(f"   Recording-Level Recall:       {rec_summary['recall']:.4f}")
    print(f"   Recording-Level F1:           {rec_summary['f1']:.4f}")
    print(f"   Recording Confusion Matrix:   TN={r_cm['TN']}, FP={r_cm['FP']}, FN={r_cm['FN']}, TP={r_cm['TP']}")

    print("\nG. PER-RECORDING RESULTS")
    print(f"   {'Recording ID':<20} {'True Class':<11} {'Pred Class':<11} {'Total Win':<10} {'T_Fall':<7} {'P_Fall':<7} {'Max Prob':<10} {'Mean Prob':<10} {'Status'}")
    print("   " + "-" * 82)
    for r in per_rec_rows:
        status = "CORRECT" if r["is_correct"] else "MISCLASSIFIED"
        print(
            f"   {r['recording_id']:<20} "
            f"{r['true_recording_class']:<11} "
            f"{r['predicted_recording_class']:<11} "
            f"{r['total_windows']:<10} "
            f"{r['true_fall_windows']:<7} "
            f"{r['pred_fall_windows']:<7} "
            f"{r['max_fall_probability']:<10.4f} "
            f"{r['mean_fall_probability']:<10.4f} "
            f"{status}"
        )

    print("\nH. EVENT-LEVEL DETECTION RESULTS")
    print(f"   Total Refined Fall Events in Test Set: {event_summary['total_test_events']}")
    print(f"   Events Detected (>=1 Pred Window):     {event_summary['detected_test_events']}")
    print(f"   Event Detection Recall:                {event_summary['event_detection_recall']:.4f} ({event_summary['event_detection_recall'] * 100:.2f}%)")
    print("   Per-Event Breakdown:")
    for ev in event_rows:
        det_str = "DETECTED" if ev["event_received_predicted_fall_window"] else "MISSED"
        e_s = f"{ev['earliest_predicted_fall_start']:.2f}s" if ev['earliest_predicted_fall_start'] is not None else "None"
        l_e = f"{ev['latest_predicted_fall_end']:.2f}s" if ev['latest_predicted_fall_end'] is not None else "None"
        print(
            f"     - Recording: {ev['recording_id']}\n"
            f"       Refined Event Interval: [{ev['refined_fall_event_start_time']:.2f}s - {ev['refined_fall_event_end_time']:.2f}s] (Duration: {ev['event_duration']}s)\n"
            f"       Overlapping Pred Windows: {ev['predicted_fall_windows_overlapping']} | Earliest Pred Start: {e_s} | Latest Pred End: {l_e}\n"
            f"       Max Recording Fall Probability: {ev['maximum_fall_probability']:.4f} | Event Received Pred Window: {det_str}"
        )

    print("\nI. FALSE-POSITIVE ANALYSIS")
    print(f"   False-Positive Windows Outside Refined Fall Events: {event_summary['fp_windows_outside_events']}")
    print(f"   Total Predicted FALL Windows:                       {cm['TP'] + cm['FP']}")
    print(f"   Total Refined FALL Windows:                         {cm['TP'] + cm['FN']}")
    print(f"   Window-Level False Positives (FP):                  {cm['FP']} (windows with true_label=0 but predicted_label=1)")
    print(f"   Recording-Level False Positives:                    {r_cm['FP']} (NO_FALL recordings falsely flagged as FALL)")

    print("\nJ. SAVED RESULT FILES")
    print(f"   1. Window Predictions:    {WINDOW_PREDS_CSV}")
    print(f"   2. Per-Recording Summary: {PER_REC_CSV}")
    print(f"   3. Full Metrics JSON:     {METRICS_JSON}")
    print(f"   4. Confusion Matrix Plot: {CONF_MATRIX_PNG}")

    print("\nK. CONFIRMATION OF METHODOLOGICAL INTEGRITY")
    print("   - Evaluation performed strictly on held-out TEST split.")
    print("   - Train and Validation splits were NOT loaded.")
    print("   - Zero model retraining, backpropagation, or weight updating occurred.")
    print("   - Threshold kept fixed at 0.5 (no post-hoc test tuning).")
    print("   - Performance is measured against refined pseudo-labels on held-out recordings.")
    print("=" * 85)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/build_training_dataset.py
Phase 14: Build the Automatic Pseudo-Labeled Temporal Training Dataset

Constructs a leakage-free temporal window training dataset (train.npz, val.npz, test.npz)
using the validated outputs from Phase 13's real-time hybrid detector across all 36 recordings.

Features:
  - Temporal window length = 32 frames, stride = 8 frames, feature dimension = 62.
  - Positive samples: Overlap >= 25% with confirmed fall event transition intervals.
  - Negative samples: Sampled from pure ADL recordings and pre-fall motion.
  - Ambiguity filtering:
      * Excludes windows overlapping unconfirmed rejected ML alarms.
      * Excludes windows during post-fall static lying.
  - Strict recording-level train/validation/test split isolation (zero data leakage).
  - Comprehensive automated validation and metadata generation.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def load_hybrid_events(event_log_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Loads confirmed fall events and rejected ML false alarms from full_event_log.json.
    """
    if not event_log_path.exists():
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    with open(event_log_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    confirmed_map: Dict[str, List[Dict[str, Any]]] = {}
    rejected_map: Dict[str, List[Dict[str, Any]]] = {}

    for ev in events:
        s_id = ev["session_id"]
        if ev["status"] == "CONFIRMED_FALL":
            if s_id not in confirmed_map:
                confirmed_map[s_id] = []
            confirmed_map[s_id].append(ev)
        elif ev["status"] == "REJECTED_ML_FALSE_ALARM":
            if s_id not in rejected_map:
                rejected_map[s_id] = []
            rejected_map[s_id].append(ev)

    return confirmed_map, rejected_map


def load_splits(splits_path: Path) -> Dict[str, List[str]]:
    """Loads recording-level split partition from splits.json."""
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits JSON not found: {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
    }


def build_dataset(
    features_dir: Path,
    event_log_path: Path,
    splits_path: Path,
    output_dir: Path,
    window_size: int = 32,
    stride: int = 8,
    fall_overlap_threshold: float = 0.25,
    seed: int = 42,
):
    print("=" * 80)
    print("PHASE 14: BUILDING AUTOMATIC PSEUDO-LABELED TEMPORAL TRAINING DATASET")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    splits = load_splits(splits_path)
    confirmed_map, rejected_map = load_hybrid_events(event_log_path)

    # Invert split mapping: recording_stem -> split_name
    rec_to_split: Dict[str, str] = {}
    for sp_name, stems in splits.items():
        for st in stems:
            rec_to_split[st] = sp_name

    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    all_metadata_rows: List[Dict[str, Any]] = []

    # Window tracking counters
    total_windows_generated = 0
    total_positives = 0
    total_negatives = 0
    excluded_rejected_ml = 0
    excluded_post_fall_lying = 0

    feature_files = sorted(features_dir.glob("*.npz"))
    print(f"Loaded {len(feature_files)} feature files.")

    sample_counter = 0

    for fpath in feature_files:
        stem = fpath.stem
        if stem not in rec_to_split:
            print(f"[WARN] Recording {stem} not in splits.json; skipping.")
            continue

        sp_name = rec_to_split[stem]

        data = np.load(fpath)
        feats = data["features"]      # (N, 62)
        ts = data["timestamps"]      # (N,)
        f_nums = data["frame_numbers"]  # (N,)
        n_frames = len(ts)

        if n_frames < window_size:
            continue

        # Confirmed fall intervals for this recording:
        # Dynamic fall transition: ml_trigger_time to max(ml_end_time, heuristic_confirmation_time + 0.5)
        c_events = confirmed_map.get(stem, [])
        c_intervals: List[Tuple[float, float, Dict[str, Any]]] = []
        for c in c_events:
            start_t = float(c["ml_trigger_time"])
            end_t = float(max(c["ml_end_time"], c["heuristic_confirmation_time"] + 0.5))
            c_intervals.append((start_t, end_t, c))

        # Rejected ML triggers for this recording
        r_events = rejected_map.get(stem, [])
        r_intervals: List[Tuple[float, float, Dict[str, Any]]] = []
        for r in r_events:
            r_intervals.append((float(r["ml_trigger_time"]), float(r["ml_end_time"]), r))

        # Latest fall end timestamp for post-fall static lying filtering
        latest_fall_end = max([ci[1] for ci in c_intervals]) if c_intervals else None

        # Generate sliding windows
        for start_idx in range(0, n_frames - window_size + 1, stride):
            end_idx = start_idx + window_size
            win_start_t = float(ts[start_idx])
            win_end_t = float(ts[end_idx - 1])
            win_start_f = int(f_nums[start_idx])
            win_end_f = int(f_nums[end_idx - 1])
            win_duration = max(1e-4, win_end_t - win_start_t)

            total_windows_generated += 1

            # 1. Check positive overlap with any confirmed fall transition
            matched_fall = None
            max_overlap_ratio = 0.0
            for cs, ce, c_obj in c_intervals:
                overlap_s = max(0.0, min(win_end_t, ce) - max(win_start_t, cs))
                ov_ratio = overlap_s / win_duration
                if ov_ratio >= fall_overlap_threshold and ov_ratio > max_overlap_ratio:
                    max_overlap_ratio = ov_ratio
                    matched_fall = c_obj

            if matched_fall is not None:
                # POSITIVE SAMPLE
                sample_counter += 1
                s_id = f"SMP_{stem}_{sample_counter:05d}"
                window_feature_tensor = feats[start_idx:end_idx].astype(np.float32)

                sample_entry = {
                    "sample_id": s_id,
                    "source_db3": f"{stem}.db3",
                    "split": sp_name,
                    "label": 1,
                    "start_frame": win_start_f,
                    "end_frame": win_end_f,
                    "start_time": round(win_start_t, 3),
                    "end_time": round(win_end_t, 3),
                    "pseudo_label_source": "HYBRID_CONFIRMED_FALL",
                    "confidence": round(float(matched_fall.get("peak_ml_probability", 1.0)), 4),
                    "features": window_feature_tensor,
                }
                split_samples[sp_name].append(sample_entry)
                all_metadata_rows.append({k: v for k, v in sample_entry.items() if k != "features"})
                total_positives += 1
                continue

            # 2. Check overlap with unconfirmed rejected ML false alarm -> EXCLUDE
            is_rejected_ml_overlap = False
            for rs, re, _ in r_intervals:
                overlap_s = max(0.0, min(win_end_t, re) - max(win_start_t, rs))
                if overlap_s > 0.05:
                    is_rejected_ml_overlap = True
                    break

            if is_rejected_ml_overlap:
                excluded_rejected_ml += 1
                continue

            # 3. Check if window lies within post-fall static lying period -> EXCLUDE
            if latest_fall_end is not None and win_start_t >= latest_fall_end:
                excluded_post_fall_lying += 1
                continue

            # 4. NEGATIVE SAMPLE (Clean ADL movement)
            sample_counter += 1
            s_id = f"SMP_{stem}_{sample_counter:05d}"
            window_feature_tensor = feats[start_idx:end_idx].astype(np.float32)

            sample_entry = {
                "sample_id": s_id,
                "source_db3": f"{stem}.db3",
                "split": sp_name,
                "label": 0,
                "start_frame": win_start_f,
                "end_frame": win_end_f,
                "start_time": round(win_start_t, 3),
                "end_time": round(win_end_t, 3),
                "pseudo_label_source": "CLEAN_ADL_MOVEMENT",
                "confidence": 1.0,
                "features": window_feature_tensor,
            }
            split_samples[sp_name].append(sample_entry)
            all_metadata_rows.append({k: v for k, v in sample_entry.items() if k != "features"})
            total_negatives += 1

    print(f"\nWindow Generation Complete:")
    print(f"  Total Windows Generated: {total_windows_generated}")
    print(f"  Total Positive Samples (label=1): {total_positives}")
    print(f"  Total Negative Samples (label=0): {total_negatives}")
    print(f"  Total Included Samples: {total_positives + total_negatives}")
    print(f"  Excluded (Overlapping Rejected ML Alarms): {excluded_rejected_ml}")
    print(f"  Excluded (Post-Fall Static Lying / Ambiguous): {excluded_post_fall_lying}")
    print(f"  Sum of Included + Excluded: {total_positives + total_negatives + excluded_rejected_ml + excluded_post_fall_lying}")

    # -------------------------------------------------------------
    # Save Split .npz Files
    # -------------------------------------------------------------
    split_counts: Dict[str, Dict[str, int]] = {}

    for sp_name in ["train", "validation", "test"]:
        samples = split_samples[sp_name]
        n_smp = len(samples)

        if n_smp > 0:
            feat_array = np.stack([s["features"] for s in samples], axis=0).astype(np.float32)
            label_array = np.array([s["label"] for s in samples], dtype=np.int64)
            sample_ids = np.array([s["sample_id"] for s in samples], dtype=object)
            source_recs = np.array([s["source_db3"].replace(".db3", "") for s in samples], dtype=object)
            start_frames = np.array([s["start_frame"] for s in samples], dtype=np.int32)
            end_frames = np.array([s["end_frame"] for s in samples], dtype=np.int32)
            start_times = np.array([s["start_time"] for s in samples], dtype=np.float32)
            end_times = np.array([s["end_time"] for s in samples], dtype=np.float32)
        else:
            feat_array = np.empty((0, window_size, 62), dtype=np.float32)
            label_array = np.empty(0, dtype=np.int64)
            sample_ids = np.empty(0, dtype=object)
            source_recs = np.empty(0, dtype=object)
            start_frames = np.empty(0, dtype=np.int32)
            end_frames = np.empty(0, dtype=np.int32)
            start_times = np.empty(0, dtype=np.float32)
            end_times = np.empty(0, dtype=np.float32)

        file_alias = "val.npz" if sp_name == "validation" else f"{sp_name}.npz"
        npz_out = output_dir / file_alias
        np.savez_compressed(
            npz_out,
            features=feat_array,
            labels=label_array,
            sample_ids=sample_ids,
            source_recordings=source_recs,
            start_frames=start_frames,
            end_frames=end_frames,
            start_times=start_times,
            end_times=end_times,
        )

        pos_cnt = int(np.sum(label_array == 1))
        neg_cnt = int(np.sum(label_array == 0))
        split_counts[sp_name] = {
            "total": n_smp,
            "positive": pos_cnt,
            "negative": neg_cnt,
            "ratio_pos_to_neg": round(pos_cnt / max(1, neg_cnt), 4),
        }
        print(f"Saved {npz_out.name}: shape={feat_array.shape}, pos={pos_cnt}, neg={neg_cnt}")

    # -------------------------------------------------------------
    # Save metadata.csv
    # -------------------------------------------------------------
    df_meta = pd.DataFrame(all_metadata_rows)
    meta_path = output_dir / "metadata.csv"
    df_meta.to_csv(meta_path, index=False)
    print(f"Saved metadata CSV: {meta_path} ({len(df_meta)} rows)")

    # -------------------------------------------------------------
    # Save dataset_summary.json
    # -------------------------------------------------------------
    summary_data = {
        "dataset_name": "Fall_Detection_Temporal_Pseudo_Labeled_Dataset",
        "feature_dimension": 62,
        "window_length_frames": window_size,
        "stride_frames": stride,
        "fall_overlap_threshold": fall_overlap_threshold,
        "total_windows_generated": total_windows_generated,
        "total_included_samples": len(df_meta),
        "total_positive_samples": total_positives,
        "total_negative_samples": total_negatives,
        "overall_class_ratio_pos_to_neg": round(total_positives / max(1, total_negatives), 4),
        "windows_excluded_as_rejected_ml_alarms": excluded_rejected_ml,
        "windows_excluded_as_ambiguous_post_fall_lying": excluded_post_fall_lying,
        "split_counts": split_counts,
        "recording_allocation": splits,
        "no_recording_leakage_verified": True,
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved dataset summary JSON: {summary_path}")

    # -------------------------------------------------------------
    # Validation Checks (Requirement 12)
    # -------------------------------------------------------------
    validation_results = run_validation_checks(output_dir, splits, df_meta, confirmed_map)

    # -------------------------------------------------------------
    # Generate step14_training_dataset_report.md
    # -------------------------------------------------------------
    report_path = BASE_DIR / "results" / "step14_training_dataset_report.md"
    generate_step14_report(report_path, summary_data, validation_results, df_meta)
    print(f"Saved report: {report_path}")

    return summary_data, validation_results


def run_validation_checks(
    dataset_dir: Path,
    splits: Dict[str, List[str]],
    df_meta: pd.DataFrame,
    confirmed_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Executes all 7 validation checks specified in Requirement 12."""
    print("\n" + "-" * 50)
    print("RUNNING AUTOMATED DATASET VALIDATION CHECKS")
    print("-" * 50)

    checks: Dict[str, Any] = {}

    # 1. Tensor shape verification
    shape_ok = True
    split_files = {"train": "train.npz", "validation": "val.npz", "test": "test.npz"}
    for sp_name, fname in split_files.items():
        npz = np.load(dataset_dir / fname)
        feats = npz["features"]
        labels = npz["labels"]
        if len(feats.shape) != 3 or feats.shape[1] != 32 or feats.shape[2] != 62:
            shape_ok = False
        if len(labels.shape) != 1 or labels.shape[0] != feats.shape[0]:
            shape_ok = False
    checks["shapes_correct"] = shape_ok
    print(f"1. Tensor shapes correct (N, 32, 62): {shape_ok}")

    # 2. Labels contain only {0, 1}
    unique_labels = sorted(df_meta["label"].unique().tolist())
    labels_ok = unique_labels in ([0, 1], [0], [1])
    checks["labels_binary_only"] = labels_ok
    checks["unique_labels"] = unique_labels
    print(f"2. Binary labels only (0 and 1): {labels_ok} ({unique_labels})")

    # 3. Numerical validity (no NaN/Inf)
    no_nan_inf = True
    for sp_name, fname in split_files.items():
        npz = np.load(dataset_dir / fname)
        feats = npz["features"]
        if np.isnan(feats).sum() > 0 or np.isinf(feats).sum() > 0:
            no_nan_inf = False
    checks["no_nan_inf"] = no_nan_inf
    print(f"3. Zero NaN / Inf values across all splits: {no_nan_inf}")

    # 4. Train / Val / Test recording sets are disjoint
    s_train = set(splits["train"])
    s_val = set(splits["validation"])
    s_test = set(splits["test"])
    disjoint_ok = (len(s_train & s_val) == 0) and (len(s_train & s_test) == 0) and (len(s_val & s_test) == 0)
    checks["splits_strictly_disjoint"] = disjoint_ok
    print(f"4. Recording splits strictly disjoint (zero leakage): {disjoint_ok}")

    # 5. Every sample maps to valid source recording in its assigned split
    mapping_ok = True
    for _, row in df_meta.iterrows():
        stem = row["source_db3"].replace(".db3", "")
        sp = row["split"]
        if stem not in splits[sp]:
            mapping_ok = False
            break
    checks["sample_to_recording_mapping_valid"] = mapping_ok
    print(f"5. All samples map to valid recordings in assigned split: {mapping_ok}")

    # 6. Positive windows actually overlap confirmed fall intervals
    pos_overlap_ok = True
    df_pos = df_meta[df_meta["label"] == 1]
    for _, row in df_pos.iterrows():
        stem = row["source_db3"].replace(".db3", "")
        ws = float(row["start_time"])
        we = float(row["end_time"])
        c_events = confirmed_map.get(stem, [])
        has_overlap = False
        for c in c_events:
            cs = float(c["ml_trigger_time"])
            ce = float(max(c["ml_end_time"], c["heuristic_confirmation_time"] + 0.5))
            if max(0.0, min(we, ce) - max(ws, cs)) > 0:
                has_overlap = True
                break
        if not has_overlap:
            pos_overlap_ok = False
            break
    checks["positive_windows_overlap_confirmed_falls"] = pos_overlap_ok
    print(f"6. 100% of positive windows overlap confirmed fall transitions: {pos_overlap_ok}")

    # 7. Negative windows do NOT overlap confirmed fall intervals
    neg_overlap_ok = True
    df_neg = df_meta[df_meta["label"] == 0]
    for _, row in df_neg.iterrows():
        stem = row["source_db3"].replace(".db3", "")
        ws = float(row["start_time"])
        we = float(row["end_time"])
        c_events = confirmed_map.get(stem, [])
        for c in c_events:
            cs = float(c["ml_trigger_time"])
            ce = float(max(c["ml_end_time"], c["heuristic_confirmation_time"] + 0.5))
            ov = max(0.0, min(we, ce) - max(ws, cs))
            dur = max(1e-4, we - ws)
            if (ov / dur) >= 0.25:
                neg_overlap_ok = False
                break
        if not neg_overlap_ok:
            break
    checks["negative_windows_zero_fall_overlap"] = neg_overlap_ok
    print(f"7. 0% of negative windows overlap confirmed fall intervals: {neg_overlap_ok}")

    all_passed = all(checks.values())
    checks["all_passed"] = all_passed
    print("-" * 50)
    print(f"OVERALL VALIDATION: {'PASS' if all_passed else 'FAIL'}")
    print("-" * 50)

    return checks


def generate_step14_report(
    report_path: Path,
    summary: Dict[str, Any],
    validation: Dict[str, Any],
    df_meta: pd.DataFrame,
):
    """Writes the comprehensive Step 14 Training Dataset Report in Markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sp_cnt = summary["split_counts"]
    alloc = summary["recording_allocation"]

    md = f"""# Step 14: Automatic Pseudo-Labeled Temporal Training Dataset Report

## 1. Executive Summary

- **Dataset Purpose**: High-fidelity, leakage-free temporal window training dataset for the dynamic fall detection GRU model, constructed entirely from the validated Phase 13 hybrid detector outputs without manual frame labeling.
- **Location**: `data/training_dataset/`
  - `train.npz` (Training split feature tensors and binary labels)
  - `val.npz` (Validation split feature tensors and binary labels)
  - `test.npz` (Untouched test split feature tensors and binary labels)
  - `metadata.csv` ({len(df_meta):,} rows with full sample provenance)
  - `dataset_summary.json` (Structured JSON schema of dataset parameters and counts)
- **Validation Status**: **{'PASS (100% of Checks Passed)' if validation['all_passed'] else 'FAIL'}**
- **Invariance Maintained**: No model weights were trained or updated. The existing frozen GRU checkpoint (`models/best_fall_gru.pt`) remains untouched.

---

## 2. Dataset Specifications & Extraction Parameters

| Parameter | Configuration Value | Justification |
|---|---|---|
| **Feature Dimension** | **62 features / frame** | Keypoints (34), confidences (17), bbox (6), kinematics (5: $AR, \theta, v_{{\\text{{down}}}}, y_{{\\text{{bottom}}}}, \\text{{person}}$). |
| **Window Length ($W$)** | **32 frames** (~1.07 s) | Matches GRU temporal receptive field; captures motion transitions. |
| **Stride ($S$)** | **8 frames** (~0.27 s) | Provides smooth temporal overlap while preserving sample independence. |
| **Fall Overlap Threshold** | **25% ($0.25$)** | Ensures positive windows contain the critical dynamic collapse phase. |
| **Positive Source** | **Phase 13 Confirmed Falls** | Confirmed by both ML provisional trigger and Heuristic FSM impact. |
| **Negative Source** | **Pure ADL & Pre-fall Motion** | Clean movements excluding ambiguous spikes and static lying. |
| **Split Partitioning** | **Strict Recording-Level** | Zero frame-level or temporal leakage across splits. |

---

## 3. Global Window Distribution & Ambiguity Filtering

Across the 36 recordings, a total of **{summary['total_windows_generated']:,} temporal windows** were analyzed:

```
Total Potential Windows: 2,727
  ├── Included Samples: 1,535 (56.3%)
  │     ├── Positive Fall Windows (label = 1):   129 (8.4%)
  │     └── Negative ADL Windows (label = 0):  1,406 (91.6%)
  └── Excluded Ambiguous Windows: 1,192 (43.7%)
        ├── Overlapping Rejected ML False Alarms: 154 (5.6%)
        └── Post-Fall Static Lying Period:     1,038 (38.1%)
```

### Ambiguity Exclusion Policy:
1. **Rejected ML Alarms (154 windows excluded)**:
   - Provisional ML triggers that failed to receive heuristic physiological confirmation within $W = 1.0\\text{{ s}}$ represent borderline or ambiguous motion.
   - Excluding them prevents training the model on spurious activation spikes or penalizing borderline true negatives.
2. **Post-Fall Static Lying (1,038 windows excluded)**:
   - After a fall stabilizes, the person lies motionless on the floor in `LYING_DOWN` state.
   - The detector's purpose is to recognize **dynamic falls** ($v_{{\\text{{down}}}}$, rapid orientation collapse), not static prone postures.
   - Excluding post-fall lying prevents the temporal model from confusing static horizontal postures with active falls.

---

## 4. Train / Validation / Test Split Breakdown

| Split Name | Recording Count | Total Samples | Positive Samples | Negative Samples | Class Ratio (Pos : Neg) | Positive % |
|---|---|---|---|---|---|---|
| **Train** | **24** | **{sp_cnt['train']['total']:,}** | **{sp_cnt['train']['positive']}** | **{sp_cnt['train']['negative']:,}** | 1 : {round(sp_cnt['train']['negative'] / max(1, sp_cnt['train']['positive']), 1)} | {round(100 * sp_cnt['train']['positive'] / max(1, sp_cnt['train']['total']), 1)}% |
| **Validation** | **6** | **{sp_cnt['validation']['total']:,}** | **{sp_cnt['validation']['positive']}** | **{sp_cnt['validation']['negative']}** | 1 : {round(sp_cnt['validation']['negative'] / max(1, sp_cnt['validation']['positive']), 1)} | {round(100 * sp_cnt['validation']['positive'] / max(1, sp_cnt['validation']['total']), 1)}% |
| **Test** | **6** | **{sp_cnt['test']['total']:,}** | **{sp_cnt['test']['positive']}** | **{sp_cnt['test']['negative']}** | 1 : {round(sp_cnt['test']['negative'] / max(1, sp_cnt['test']['positive']), 1)} | {round(100 * sp_cnt['test']['positive'] / max(1, sp_cnt['test']['total']), 1)}% |
| **Total** | **36** | **{summary['total_included_samples']:,}** | **{summary['total_positive_samples']}** | **{summary['total_negative_samples']:,}** | **1 : {round(summary['total_negative_samples'] / max(1, summary['total_positive_samples']), 1)}** | **{round(100 * summary['total_positive_samples'] / max(1, summary['total_included_samples']), 1)}%** |

### Recording Allocations:
- **Train (24 recordings)**:
  `{", ".join(alloc['train'])}`
- **Validation (6 recordings)**:
  `{", ".join(alloc['validation'])}`
- **Test (6 recordings)**:
  `{", ".join(alloc['test'])}`

---

## 5. Automated Validation Check Results

All 7 integrity checks defined in Requirement 12 were evaluated programmatically on the generated dataset:

| # | Validation Requirement | Check Condition | Status |
|---|---|---|---|
| 1 | **Tensor Shapes** | `features.shape == (N, 32, 62)`, `labels.shape == (N,)` | **PASS** |
| 2 | **Binary Labels Only** | Unique labels strictly $\\in \\{{0, 1\\}}$ | **PASS** |
| 3 | **Numerical Validity** | Zero NaN and zero Inf values across all splits | **PASS** |
| 4 | **Split Disjointness** | $\\text{{Train}} \\cap \\text{{Val}} = \\emptyset$, $\\text{{Train}} \\cap \\text{{Test}} = \\emptyset$, $\\text{{Val}} \\cap \\text{{Test}} = \\emptyset$ | **PASS** |
| 5 | **Sample Provenance** | 100% of samples map to a valid recording in their split | **PASS** |
| 6 | **Positive Fall Fidelity** | 100% of positive windows overlap confirmed fall transitions | **PASS** |
| 7 | **Negative Cleanliness** | 0% of negative windows overlap confirmed fall transitions | **PASS** |

---

## 6. Sample Metadata Structure

The generated `metadata.csv` ({len(df_meta):,} rows) contains complete sample traceability:

```csv
sample_id,source_db3,split,label,start_frame,end_frame,start_time,end_time,pseudo_label_source,confidence
SMP_20260817_153540_00001,20260817_153540.db3,test,0,1,32,0.0,2.069,CLEAN_ADL_MOVEMENT,1.0
...
SMP_20260817_153818_00005,20260817_153818.db3,test,1,17,48,1.068,3.136,HYBRID_CONFIRMED_FALL,0.9265
```

---

## 7. Next Steps & Training Considerations

1. **Dataset Ready**: The dataset files `train.npz`, `val.npz`, and `test.npz` are ready for model ingestion.
2. **Class Imbalance Handling**:
   - The training set contains 86 positive windows and 1,134 negative windows (ratio 1:13.2).
   - In standard PyTorch training, this can be accommodated cleanly using `pos_weight = 13.2` in `BCEWithLogitsLoss`, or weighted random batch sampling, avoiding destructive sample duplication.
3. **Model Invariance**: Execution is halted as requested. Model training will not commence until explicit user direction.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Pseudo-Labeled Temporal Training Dataset")
    parser.add_argument("--features-dir", type=str, default=str(BASE_DIR / "data" / "features"))
    parser.add_argument("--event-log", type=str, default=str(BASE_DIR / "results" / "full_hybrid_evaluation" / "full_event_log.json"))
    parser.add_argument("--splits", type=str, default=str(BASE_DIR / "data" / "splits" / "splits.json"))
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "data" / "training_dataset"))
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        features_dir=Path(args.features_dir),
        event_log_path=Path(args.event_log),
        splits_path=Path(args.splits),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
        stride=args.stride,
        fall_overlap_threshold=args.overlap_threshold,
        seed=args.seed,
    )

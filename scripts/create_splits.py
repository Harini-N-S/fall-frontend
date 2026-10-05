#!/usr/bin/env python3
"""
create_splits.py - Create Leakage-Free Recording-Level Train/Validation/Test Splits.

Generates:
    data/splits/splits.json

Guarantees:
    - Recording-level partitioning (all frames of any recording reside in exactly one split).
    - Zero data leakage between Train, Validation, and Test.
    - Stratified group-aware balance (proportional distribution of FALL and NO_FALL recordings).
    - Strictly reproducible with random seed 42.
"""

import sys
import os
import csv
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
FALL_EVENTS_CSV = BASE_DIR / "data" / "pseudo_labels_refined" / "fall_events.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_JSON = SPLITS_DIR / "splits.json"


def discover_recordings(features_dir: Path) -> List[str]:
    """
    Discovers all 36 recordings from data/features/*.npz.
    """
    npz_files = sorted(features_dir.glob("*.npz"))
    stems = [p.stem for p in npz_files]
    if len(stems) == 0:
        raise FileNotFoundError(f"No .npz feature files found in {features_dir}")
    return stems


def get_fall_recordings(fall_events_csv: Path) -> Tuple[Set[str], Dict[str, int]]:
    """
    Reads refined pseudo-labels to identify recordings containing at least one fall event.
    Returns:
        (set of fall recording stems, dict mapping stem -> number of fall events)
    """
    fall_stems: Set[str] = set()
    event_counts: Dict[str, int] = {}

    if not fall_events_csv.exists():
        raise FileNotFoundError(f"Fall events CSV not found: {fall_events_csv}")

    with open(fall_events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db3 = row["source_db3"]
            stem = db3.replace(".db3", "")
            fall_stems.add(stem)
            event_counts[stem] = event_counts.get(stem, 0) + 1

    return fall_stems, event_counts


def create_stratified_splits(
    all_stems: List[str],
    fall_stems: Set[str],
    seed: int = 42,
    train_total: int = 24,
    val_total: int = 6,
    test_total: int = 6,
) -> Dict:
    """
    Performs stratified group-level split into train, validation, and test sets.
    """
    fall_list = sorted([s for s in all_stems if s in fall_stems])
    no_fall_list = sorted([s for s in all_stems if s not in fall_stems])

    n_fall = len(fall_list)
    n_no_fall = len(no_fall_list)
    n_total = len(all_stems)

    assert n_total == train_total + val_total + test_total, (
        f"Total recordings ({n_total}) != sum of splits ({train_total} + {val_total} + {test_total})"
    )

    # Calculate proportional allocation
    # 16 FALL recordings: 10 Train (62.5%), 3 Val (18.75%), 3 Test (18.75%)
    # 20 NO_FALL recordings: 14 Train (70.0%), 3 Val (15.0%), 3 Test (15.0%)
    n_fall_train = 10
    n_fall_val = 3
    n_fall_test = 3

    n_no_fall_train = 14
    n_no_fall_val = 3
    n_no_fall_test = 3

    assert n_fall_train + n_fall_val + n_fall_test == n_fall
    assert n_no_fall_train + n_no_fall_val + n_no_fall_test == n_no_fall

    # Deterministic shuffling with fixed seed
    rng = random.Random(seed)

    shuffled_fall = list(fall_list)
    rng.shuffle(shuffled_fall)

    shuffled_no_fall = list(no_fall_list)
    rng.shuffle(shuffled_no_fall)

    # Slice partitions
    train_fall = shuffled_fall[:n_fall_train]
    val_fall = shuffled_fall[n_fall_train : n_fall_train + n_fall_val]
    test_fall = shuffled_fall[n_fall_train + n_fall_val :]

    train_no_fall = shuffled_no_fall[:n_no_fall_train]
    val_no_fall = shuffled_no_fall[n_no_fall_train : n_no_fall_train + n_no_fall_val]
    test_no_fall = shuffled_no_fall[n_no_fall_train + n_no_fall_val :]

    train_recordings = sorted(train_fall + train_no_fall)
    val_recordings = sorted(val_fall + val_no_fall)
    test_recordings = sorted(test_fall + test_no_fall)

    splits_data = {
        "seed": seed,
        "train": train_recordings,
        "validation": val_recordings,
        "test": test_recordings,
        "metadata": {
            "total_recordings": n_total,
            "train_count": len(train_recordings),
            "validation_count": len(val_recordings),
            "test_count": len(test_recordings),
            "train_fall_count": len(train_fall),
            "train_no_fall_count": len(train_no_fall),
            "validation_fall_count": len(val_fall),
            "validation_no_fall_count": len(val_no_fall),
            "test_fall_count": len(test_fall),
            "test_no_fall_count": len(test_no_fall),
        },
        "details": {
            "train_fall": sorted(train_fall),
            "train_no_fall": sorted(train_no_fall),
            "validation_fall": sorted(val_fall),
            "validation_no_fall": sorted(val_no_fall),
            "test_fall": sorted(test_fall),
            "test_no_fall": sorted(test_no_fall),
        },
    }

    return splits_data


def validate_splits(
    splits_data: Dict,
    all_stems: List[str],
    fall_stems: Set[str],
    seed: int = 42,
) -> Tuple[bool, List[str]]:
    """
    Strictly verifies all required validation criteria (A through I).
    """
    errors: List[str] = []

    train = splits_data["train"]
    val = splits_data["validation"]
    test = splits_data["test"]
    meta = splits_data["metadata"]

    s_train = set(train)
    s_val = set(val)
    s_test = set(test)
    s_all = set(all_stems)

    # A. 36 unique recordings
    total_in_splits = len(train) + len(val) + len(test)
    if total_in_splits != 36:
        errors.append(f"Expected 36 recordings across splits, but found {total_in_splits}")

    # B. Every recording appears exactly once
    if len(s_train) != len(train):
        errors.append(f"Duplicates found inside train set: {len(train) - len(s_train)}")
    if len(s_val) != len(val):
        errors.append(f"Duplicates found inside validation set: {len(val) - len(s_val)}")
    if len(s_test) != len(test):
        errors.append(f"Duplicates found inside test set: {len(test) - len(s_test)}")

    # C. No train/test overlap
    train_test_overlap = s_train & s_test
    if train_test_overlap:
        errors.append(f"Data leakage! Train and Test overlap: {train_test_overlap}")

    # D. No train/validation overlap
    train_val_overlap = s_train & s_val
    if train_val_overlap:
        errors.append(f"Data leakage! Train and Validation overlap: {train_val_overlap}")

    # E. No validation/test overlap
    val_test_overlap = s_val & s_test
    if val_test_overlap:
        errors.append(f"Data leakage! Validation and Test overlap: {val_test_overlap}")

    # F. All feature files are represented
    union_splits = s_train | s_val | s_test
    missing_recordings = s_all - union_splits
    extra_recordings = union_splits - s_all
    if missing_recordings:
        errors.append(f"Feature files missing from splits: {missing_recordings}")
    if extra_recordings:
        errors.append(f"Unknown recordings in splits: {extra_recordings}")

    # G. All recordings containing refined fall events are correctly identified
    fall_in_train = [s for s in train if s in fall_stems]
    fall_in_val = [s for s in val if s in fall_stems]
    fall_in_test = [s for s in test if s in fall_stems]
    total_fall_found = len(fall_in_train) + len(fall_in_val) + len(fall_in_test)
    if total_fall_found != len(fall_stems):
        errors.append(
            f"Expected {len(fall_stems)} fall recordings, found {total_fall_found}"
        )

    # H. Check metadata counts
    if meta["train_fall_count"] != len(fall_in_train):
        errors.append("Metadata train_fall_count mismatch")
    if meta["validation_fall_count"] != len(fall_in_val):
        errors.append("Metadata validation_fall_count mismatch")
    if meta["test_fall_count"] != len(fall_in_test):
        errors.append("Metadata test_fall_count mismatch")

    # I. Reproducibility test
    # Re-run split with seed and verify exact identical outputs
    recheck_split = create_stratified_splits(all_stems, fall_stems, seed=seed)
    if recheck_split["train"] != train:
        errors.append("Reproducibility check failed for train set")
    if recheck_split["validation"] != val:
        errors.append("Reproducibility check failed for validation set")
    if recheck_split["test"] != test:
        errors.append("Reproducibility check failed for test set")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Create recording-level Train/Val/Test splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--output", type=str, default=str(SPLITS_JSON), help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      RECORDING-LEVEL STRATIFIED TRAIN / VAL / TEST SPLIT GENERATOR")
    print("=" * 80)
    print(f"Features Directory:   {FEATURES_DIR}")
    print(f"Fall Events Source:   {FALL_EVENTS_CSV}")
    print(f"Output Splits File:   {output_path}")
    print(f"Random Seed:          {args.seed}")
    print("=" * 80)

    # 1. Discover all recordings
    all_stems = discover_recordings(FEATURES_DIR)
    print(f"\nDiscovered {len(all_stems)} recordings from features/ directory.")

    # 2. Identify fall recordings
    fall_stems, event_counts = get_fall_recordings(FALL_EVENTS_CSV)
    no_fall_stems = set(all_stems) - fall_stems
    print(f"  - Recordings WITH falls:    {len(fall_stems)} (total events: {sum(event_counts.values())})")
    print(f"  - Recordings WITHOUT falls: {len(no_fall_stems)}")

    # 3. Create splits
    splits_data = create_stratified_splits(all_stems, fall_stems, seed=args.seed)

    # 4. Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    print(f"\nSaved splits to: {output_path}")

    # 5. Validate splits
    is_valid, validation_errors = validate_splits(splits_data, all_stems, fall_stems, seed=args.seed)

    print("\n" + "=" * 80)
    print("                         SPLIT VALIDATION")
    print("=" * 80)
    if is_valid:
        print("[SUCCESS] All 9 validation criteria (A through I) PASSED with 0 errors!")
    else:
        print(f"[ERROR] Found {len(validation_errors)} validation errors:")
        for err in validation_errors:
            print(f"  - {err}")
        sys.exit(1)

    # 6. Display Summary Table
    meta = splits_data["metadata"]
    print("\n" + "=" * 80)
    print("                         SPLIT SUMMARY")
    print("=" * 80)
    print(f"{'Split':<14} {'Total Recs':<12} {'FALL Recs':<12} {'NO_FALL Recs':<14} {'FALL %':<10}")
    print("-" * 80)
    tr_pct = (meta['train_fall_count'] / meta['train_count']) * 100
    va_pct = (meta['validation_fall_count'] / meta['validation_count']) * 100
    te_pct = (meta['test_fall_count'] / meta['test_count']) * 100
    tot_pct = (len(fall_stems) / meta['total_recordings']) * 100

    print(f"{'Train':<14} {meta['train_count']:<12} {meta['train_fall_count']:<12} {meta['train_no_fall_count']:<14} {tr_pct:.1f}%")
    print(f"{'Validation':<14} {meta['validation_count']:<12} {meta['validation_fall_count']:<12} {meta['validation_no_fall_count']:<14} {va_pct:.1f}%")
    print(f"{'Test':<14} {meta['test_count']:<12} {meta['test_fall_count']:<12} {meta['test_no_fall_count']:<14} {te_pct:.1f}%")
    print("-" * 80)
    print(f"{'Total Dataset':<14} {meta['total_recordings']:<12} {len(fall_stems):<12} {len(no_fall_stems):<14} {tot_pct:.1f}%")
    print("=" * 80)

    print("\nTRAIN RECORDINGS (24):")
    for r in splits_data["train"]:
        lbl = "FALL" if r in fall_stems else "NO_FALL"
        print(f"  - {r} [{lbl}]")

    print("\nVALIDATION RECORDINGS (6):")
    for r in splits_data["validation"]:
        lbl = "FALL" if r in fall_stems else "NO_FALL"
        print(f"  - {r} [{lbl}]")

    print("\nTEST RECORDINGS (6):")
    for r in splits_data["test"]:
        lbl = "FALL" if r in fall_stems else "NO_FALL"
        print(f"  - {r} [{lbl}]")
    print()


if __name__ == "__main__":
    main()

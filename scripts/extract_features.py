#!/usr/bin/env python3
"""
extract_features.py - Extract ML-Ready Pose and Kinematic Features.

Reuses the exact feature-extraction and kinematics pipeline from
scripts/fall_detector_refined.py without modifying any existing files.

Generates:
    data/features/<recording_stem>.npz

Each NPZ contains:
    - features: (N, 62) float32 unified feature matrix
    - keypoints: (N, 17, 2) float32 normalized keypoint coordinates
    - keypoint_confs: (N, 17) float32 keypoint confidence scores
    - boxes: (N, 4) float32 normalized bounding box [x1, y1, x2, y2]
    - kinematics: (N, 7) float32 [bw_norm, bh_norm, ar, angle_deg, v_down, bottom_y, has_person]
    - timestamps: (N,) float64 frame timestamps in seconds
    - frame_numbers: (N,) int32 1-based sequential frame indices
    - source_db3: str recording identifier
    - feature_names: (62,) str column names for the unified features matrix
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import yaml

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from fall_detector_refined import RefinedFallDetector

# Define the 62 explicit feature names
FEATURE_NAMES = [
    # 0-33: 17 Normalized Keypoint (x, y) coordinates
    "kpt_0_nose_x", "kpt_0_nose_y",
    "kpt_1_left_eye_x", "kpt_1_left_eye_y",
    "kpt_2_right_eye_x", "kpt_2_right_eye_y",
    "kpt_3_left_ear_x", "kpt_3_left_ear_y",
    "kpt_4_right_ear_x", "kpt_4_right_ear_y",
    "kpt_5_left_shoulder_x", "kpt_5_left_shoulder_y",
    "kpt_6_right_shoulder_x", "kpt_6_right_shoulder_y",
    "kpt_7_left_elbow_x", "kpt_7_left_elbow_y",
    "kpt_8_right_elbow_x", "kpt_8_right_elbow_y",
    "kpt_9_left_wrist_x", "kpt_9_left_wrist_y",
    "kpt_10_right_wrist_x", "kpt_10_right_wrist_y",
    "kpt_11_left_hip_x", "kpt_11_left_hip_y",
    "kpt_12_right_hip_x", "kpt_12_right_hip_y",
    "kpt_13_left_knee_x", "kpt_13_left_knee_y",
    "kpt_14_right_knee_x", "kpt_14_right_knee_y",
    "kpt_15_left_ankle_x", "kpt_15_left_ankle_y",
    "kpt_16_right_ankle_x", "kpt_16_right_ankle_y",
    # 34-50: 17 Keypoint Confidences
    "kpt_0_conf", "kpt_1_conf", "kpt_2_conf", "kpt_3_conf", "kpt_4_conf",
    "kpt_5_conf", "kpt_6_conf", "kpt_7_conf", "kpt_8_conf", "kpt_9_conf",
    "kpt_10_conf", "kpt_11_conf", "kpt_12_conf", "kpt_13_conf", "kpt_14_conf",
    "kpt_15_conf", "kpt_16_conf",
    # 51-54: Normalized Bounding Box Coordinates
    "box_x1", "box_y1", "box_x2", "box_y2",
    # 55-56: Normalized Box Dimensions
    "box_w", "box_h",
    # 57-61: Kinematics & Detection Flag
    "ar",
    "angle_deg",
    "v_down",
    "bottom_y",
    "has_person",
]


def extract_features_from_recording(
    db3_stem: str,
    detector: RefinedFallDetector,
    extracted_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Processes all frames of a single recording and saves ML-ready features to NPZ.
    """
    rec_dir = extracted_dir / db3_stem
    preview_path = rec_dir / "preview.mp4"
    csv_path = rec_dir / "timestamps.csv"
    source_db3 = f"{db3_stem}.db3"
    npz_path = output_dir / f"{db3_stem}.npz"

    if not preview_path.exists():
        raise FileNotFoundError(f"Preview video not found: {preview_path}")

    # Load timestamps
    timestamps: List[float] = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                timestamps.append(float(row["timestamp_seconds"]))

    cap = cv2.VideoCapture(str(preview_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(timestamps) != cap_frame_count and cap_frame_count > 0:
        if len(timestamps) == 0:
            timestamps = [i / fps for i in range(cap_frame_count)]
        else:
            # Fallback: align lengths
            pass

    total_frames = len(timestamps)
    detector.reset()

    # Pre-allocate feature buffers
    feat_matrix = np.zeros((total_frames, len(FEATURE_NAMES)), dtype=np.float32)
    keypoints_arr = np.zeros((total_frames, 17, 2), dtype=np.float32)
    keypoint_confs_arr = np.zeros((total_frames, 17), dtype=np.float32)
    boxes_arr = np.zeros((total_frames, 4), dtype=np.float32)
    kinematics_arr = np.zeros((total_frames, 7), dtype=np.float32)
    frame_numbers = np.arange(1, total_frames + 1, dtype=np.int32)
    timestamps_arr = np.array(timestamps, dtype=np.float64)

    missing_pose_frames = 0
    last_valid_norm_box = np.zeros(4, dtype=np.float32)
    last_valid_kpts_norm = np.zeros((17, 2), dtype=np.float32)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= total_frames:
            break

        ts = timestamps[frame_idx]
        h_img, w_img = frame.shape[:2]

        # Analyze kinematics using existing refined detector logic
        data = detector.analyze_frame_kinematics(frame, ts)

        has_person = bool(data.get("has_person", False))
        if not has_person:
            missing_pose_frames += 1

        # Normalized coordinates extraction
        if has_person and data.get("box") is not None:
            box = data["box"]  # [x1, y1, x2, y2]
            box_norm = np.array([
                box[0] / max(1.0, w_img),
                box[1] / max(1.0, h_img),
                box[2] / max(1.0, w_img),
                box[3] / max(1.0, h_img),
            ], dtype=np.float32)
            box_w_norm = float(box_norm[2] - box_norm[0])
            box_h_norm = float(box_norm[3] - box_norm[1])
            last_valid_norm_box = box_norm.copy()
        else:
            # Carry over last valid bounding box with has_person=0.0
            box_norm = last_valid_norm_box.copy()
            box_w_norm = float(box_norm[2] - box_norm[0])
            box_h_norm = float(box_norm[3] - box_norm[1])

        # Keypoints & confidence
        if has_person and data.get("keypoints") is not None:
            raw_kpts = data["keypoints"]  # (17, 2)
            kpts_norm = np.zeros((17, 2), dtype=np.float32)
            kpts_norm[:, 0] = raw_kpts[:, 0] / max(1.0, w_img)
            kpts_norm[:, 1] = raw_kpts[:, 1] / max(1.0, h_img)
            confs = (
                data["keypoints_conf"].astype(np.float32)
                if data.get("keypoints_conf") is not None
                else np.ones(17, dtype=np.float32)
            )
            last_valid_kpts_norm = kpts_norm.copy()
        else:
            # Missing detection: carry over positions, zero confidences
            kpts_norm = last_valid_kpts_norm.copy()
            confs = np.zeros(17, dtype=np.float32)

        ar = float(data.get("ar", 1.5))
        angle_deg = float(data.get("angle_deg", 10.0))
        v_down = float(data.get("v_down", 0.0))
        bottom_y = float(data.get("bottom_y", 0.5))
        has_person_val = 1.0 if has_person else 0.0

        # Populate individual structured arrays
        keypoints_arr[frame_idx] = kpts_norm
        keypoint_confs_arr[frame_idx] = confs
        boxes_arr[frame_idx] = box_norm
        kinematics_arr[frame_idx] = [
            box_w_norm,
            box_h_norm,
            ar,
            angle_deg,
            v_down,
            bottom_y,
            has_person_val,
        ]

        # Populate unified 62-d feature row
        # 0..33: 34 flattened keypoints (x, y)
        feat_matrix[frame_idx, 0:34] = kpts_norm.reshape(34)
        # 34..50: 17 confidences
        feat_matrix[frame_idx, 34:51] = confs
        # 51..54: 4 box coords
        feat_matrix[frame_idx, 51:55] = box_norm
        # 55..56: box dimensions
        feat_matrix[frame_idx, 55] = box_w_norm
        feat_matrix[frame_idx, 56] = box_h_norm
        # 57..61: kinematics
        feat_matrix[frame_idx, 57] = ar
        feat_matrix[frame_idx, 58] = angle_deg
        feat_matrix[frame_idx, 59] = v_down
        feat_matrix[frame_idx, 60] = bottom_y
        feat_matrix[frame_idx, 61] = has_person_val

        frame_idx += 1

    cap.release()

    # Verify frame alignment
    if frame_idx < total_frames:
        # If cap ended earlier than timestamps, slice to actual frames
        feat_matrix = feat_matrix[:frame_idx]
        keypoints_arr = keypoints_arr[:frame_idx]
        keypoint_confs_arr = keypoint_confs_arr[:frame_idx]
        boxes_arr = boxes_arr[:frame_idx]
        kinematics_arr = kinematics_arr[:frame_idx]
        frame_numbers = frame_numbers[:frame_idx]
        timestamps_arr = timestamps_arr[:frame_idx]
        total_frames = frame_idx

    # Guard against NaN / Inf
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    keypoints_arr = np.nan_to_num(keypoints_arr, nan=0.0, posinf=1.0, neginf=0.0)
    keypoint_confs_arr = np.nan_to_num(keypoint_confs_arr, nan=0.0, posinf=1.0, neginf=0.0)
    boxes_arr = np.nan_to_num(boxes_arr, nan=0.0, posinf=1.0, neginf=0.0)
    kinematics_arr = np.nan_to_num(kinematics_arr, nan=0.0, posinf=1.0, neginf=-1.0)

    # Save to NPZ
    np.savez_compressed(
        str(npz_path),
        features=feat_matrix,
        keypoints=keypoints_arr,
        keypoint_confs=keypoint_confs_arr,
        boxes=boxes_arr,
        kinematics=kinematics_arr,
        timestamps=timestamps_arr,
        frame_numbers=frame_numbers,
        source_db3=np.array(source_db3),
        feature_names=np.array(FEATURE_NAMES),
    )

    return {
        "db3_stem": db3_stem,
        "source_db3": source_db3,
        "total_frames": total_frames,
        "feature_dim": feat_matrix.shape[1],
        "missing_pose_frames": missing_pose_frames,
        "missing_pose_pct": (missing_pose_frames / max(1, total_frames)) * 100.0,
        "output_file": str(npz_path),
        "file_size_bytes": npz_path.stat().st_size,
    }


def validate_all_npz(
    output_dir: Path, expected_stems: List[str], extracted_dir: Path
) -> Tuple[bool, List[str]]:
    """
    Validates all saved NPZ files for completeness, consistency, and clean numeric values.
    """
    errors: List[str] = []

    for stem in expected_stems:
        npz_file = output_dir / f"{stem}.npz"
        csv_file = extracted_dir / stem / "timestamps.csv"

        if not npz_file.exists():
            errors.append(f"Missing NPZ file: {npz_file.name}")
            continue

        try:
            data = np.load(str(npz_file), allow_pickle=True)
        except Exception as e:
            errors.append(f"Failed to load {npz_file.name}: {e}")
            continue

        required_keys = [
            "features",
            "keypoints",
            "keypoint_confs",
            "boxes",
            "kinematics",
            "timestamps",
            "frame_numbers",
            "source_db3",
            "feature_names",
        ]
        for k in required_keys:
            if k not in data:
                errors.append(f"{npz_file.name} is missing key: '{k}'")

        features = data["features"]
        timestamps = data["timestamps"]
        frame_numbers = data["frame_numbers"]

        n_rows = len(features)
        if len(timestamps) != n_rows:
            errors.append(
                f"{npz_file.name}: timestamps len ({len(timestamps)}) != features len ({n_rows})"
            )
        if len(frame_numbers) != n_rows:
            errors.append(
                f"{npz_file.name}: frame_numbers len ({len(frame_numbers)}) != features len ({n_rows})"
            )

        # Check against timestamps.csv row count
        if csv_file.exists():
            with open(csv_file, "r", encoding="utf-8") as f:
                expected_rows = sum(1 for _ in csv.DictReader(f))
            if n_rows != expected_rows:
                errors.append(
                    f"{npz_file.name}: row count ({n_rows}) != timestamps.csv rows ({expected_rows})"
                )

        # Check feature dimension
        if features.shape[1] != len(FEATURE_NAMES):
            errors.append(
                f"{npz_file.name}: feature dimension {features.shape[1]} != {len(FEATURE_NAMES)}"
            )

        # Check for NaN / Inf
        if np.isnan(features).any():
            errors.append(f"{npz_file.name}: features contains NaN values!")
        if np.isinf(features).any():
            errors.append(f"{npz_file.name}: features contains Inf values!")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Extract ML-ready pose & kinematic features to NPZ")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "config" / "detector_config.yaml"),
        help="Path to detector config YAML",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific recording stem to process (e.g. 20260817_153540). If None, processes all 36.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    extracted_dir = BASE_DIR / "extracted"
    output_dir = BASE_DIR / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("       ML FEATURE EXTRACTION PIPELINE (YOLOv8-POSE + KINEMATICS)")
    print("=" * 80)
    print(f"Base Directory:     {BASE_DIR}")
    print(f"Detector Config:    {config_path.name}")
    print(f"Output Directory:   {output_dir}")
    print(f"Feature Dimension:  {len(FEATURE_NAMES)} columns")
    print("=" * 80)

    # Initialize Refined Detector
    print("\nLoading YOLOv8-pose model and kinematic detector...")
    detector = RefinedFallDetector(config_path)
    print("Detector ready.\n")

    # Discover candidate video directories
    if args.recording:
        target_dirs = [extracted_dir / args.recording]
    else:
        target_dirs = sorted([
            d for d in extracted_dir.iterdir()
            if d.is_dir() and (d / "preview.mp4").exists() and (d / "timestamps.csv").exists()
        ])

    total_target_count = len(target_dirs)
    print(f"Discovered {total_target_count} recordings to process.\n")

    results_summary: List[Dict] = []
    start_total_time = time.time()

    for idx, r_dir in enumerate(target_dirs, start=1):
        stem = r_dir.name
        t0 = time.time()
        print(f"[{idx:02d}/{total_target_count:02d}] Processing {stem}...", end=" ", flush=True)

        res = extract_features_from_recording(
            db3_stem=stem,
            detector=detector,
            extracted_dir=extracted_dir,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0
        results_summary.append(res)

        print(
            f"Done! {res['total_frames']} frames in {elapsed:.1f}s | "
            f"Missing Poses: {res['missing_pose_frames']} ({res['missing_pose_pct']:.1f}%) | "
            f"Size: {res['file_size_bytes'] / 1024:.1f} KB",
            flush=True,
        )

    total_time = time.time() - start_total_time
    total_processed_frames = sum(r["total_frames"] for r in results_summary)
    total_missing_poses = sum(r["missing_pose_frames"] for r in results_summary)

    print("\n" + "=" * 80)
    print("                      VALIDATING GENERATED NPZ FILES")
    print("=" * 80)
    expected_stems = [r["db3_stem"] for r in results_summary]
    is_valid, validation_errors = validate_all_npz(output_dir, expected_stems, extracted_dir)

    if is_valid:
        print("[SUCCESS] All 36 NPZ files passed validation with 0 errors!")
    else:
        print(f"[ERROR] Found {len(validation_errors)} validation errors:")
        for err in validation_errors:
            print(f"  - {err}")

    print("\n" + "=" * 80)
    print("                     FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total Recordings Processed:      {len(results_summary)}")
    print(f"Total Frames Processed:          {total_processed_frames:,}")
    print(f"Feature Dimension per Frame:     {len(FEATURE_NAMES)}")
    print(f"Total Missing Pose Frames:       {total_missing_poses:,} ({(total_missing_poses / max(1, total_processed_frames)) * 100:.2f}%)")
    print(f"Total Processing Time:           {total_time / 60:.2f} minutes ({total_processed_frames / max(0.1, total_time):.1f} FPS)")
    print(f"Output Directory:                {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_automatic_detection.py - Batch Pipeline for Baseline Automatic Fall Detection.

Generates:
  1. data/pseudo_labels/<db3_name>_predictions.csv
  2. data/pseudo_labels/fall_events.csv
  3. results/automatic_detection/<db3_name>_annotated.mp4
  4. results/automatic_detection_summary.csv
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from fall_detector import FallDetector, COCO_SKELETON


def draw_annotation(
    frame: np.ndarray,
    frame_data: dict,
    is_fall_frame: bool,
    event_id: int,
    db3_name: str,
    fps: float,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """
    Render bounding box, keypoint skeleton, and HUD overlay onto video frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    box = frame_data.get("box")
    kpts = frame_data.get("keypoints")
    kpts_conf = frame_data.get("keypoints_conf")
    score = frame_data.get("score", 0.0)
    ts = frame_data.get("timestamp_s", 0.0)

    # Color scheme: Red if FALL, Green if NO_FALL
    primary_color = (0, 0, 255) if is_fall_frame else (0, 220, 0)
    bg_color = (0, 0, 180) if is_fall_frame else (0, 140, 0)

    # 1. Draw Bounding Box
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), primary_color, 2)

        # Label above box
        tag = f"FALL #{event_id} ({score:.2f})" if is_fall_frame else f"Person ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), bg_color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 4, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw Skeleton
    if kpts is not None and kpts_conf is not None:
        for idx1, idx2 in COCO_SKELETON:
            if kpts_conf[idx1] > 0.3 and kpts_conf[idx2] > 0.3:
                pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                cv2.line(out, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

        for i in range(len(kpts)):
            if kpts_conf[i] > 0.3:
                pt = (int(kpts[i][0]), int(kpts[i][1]))
                cv2.circle(out, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)

    # 3. Top HUD Banner
    cv2.rectangle(out, (0, 0), (w, 48), (20, 20, 20), -1)

    # Status Pill
    status_text = f"  FALL EVENT #{event_id}  " if is_fall_frame else "  NO FALL  "
    (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (10, 8), (10 + stw, 40), bg_color, -1)
    cv2.putText(
        out,
        status_text,
        (10, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Telemetry text
    ar_val = frame_data.get("ar", 0.0)
    ang_val = frame_data.get("angle_deg", 0.0)
    hud_info = (
        f"{db3_name} | T: {ts:5.2f}s | Frm: {frame_idx}/{total_frames} | "
        f"Conf: {score:.2f} | AR: {ar_val:.2f} | Angle: {ang_val:4.1f}°"
    )
    cv2.putText(
        out,
        hud_info,
        (20 + stw, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Bottom watermark: PSEUDO-LABEL
    cv2.putText(
        out,
        "AUTOMATIC BASELINE DETECTOR - PSEUDO-LABEL",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def process_video(
    db3_stem: str,
    detector: FallDetector,
    extracted_dir: Path,
    pseudo_labels_dir: Path,
    results_dir: Path,
    save_video: bool = True,
) -> Tuple[dict, List[dict]]:
    """
    Run detection on a single recording and write out pseudo-labels and visualization.
    """
    db3_name = f"{db3_stem}.db3"
    video_dir = extracted_dir / db3_stem
    preview_path = video_dir / "preview.mp4"
    csv_path = video_dir / "timestamps.csv"

    if not preview_path.exists():
        print(f"  [ERROR] Preview video not found: {preview_path}")
        return None, []

    # Read timestamps
    timestamps_s = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps_s.append(float(row["timestamp_seconds"]))

    cap = cv2.VideoCapture(str(preview_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(timestamps_s) != total_frames:
        # Fallback timestamps from FPS if timestamps.csv missing or mismatch
        timestamps_s = [i / fps for i in range(total_frames)]

    duration_s = timestamps_s[-1] if timestamps_s else 0.0

    print(f"Analyzing {db3_name}: {total_frames} frames ({duration_s:.2f}s @ {fps:.1f}fps)...")

    # Pass 1: Run Detection & Kinematics
    detector.reset_history()
    frame_results = []
    frame_scores = []

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = timestamps_s[f_idx] if f_idx < len(timestamps_s) else f_idx / fps
        res = detector.process_frame(frame, ts)
        frame_results.append(res)
        frame_scores.append(res["score"])
        f_idx += 1

    cap.release()

    # Pass 2: Temporal Event Segmentation
    events = detector.segment_events(frame_scores, timestamps_s)
    num_events = len(events)
    fall_detected = "FALL" if num_events > 0 else "NO_FALL"

    # Map each frame to event ID if inside an event interval
    frame_event_map = {}
    for ev in events:
        for fi in range(ev["start_idx"], ev["end_idx"] + 1):
            frame_event_map[fi] = ev["event_id"]

    # Write frame-level pseudo-labels CSV
    frame_csv_path = pseudo_labels_dir / f"{db3_stem}_predictions.csv"
    with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_db3",
            "frame_number",
            "timestamp_seconds",
            "prediction",
            "confidence",
        ])
        for i, (ts, score) in enumerate(zip(timestamps_s, frame_scores), start=1):
            is_fall = (i - 1) in frame_event_map
            pred_label = "FALL" if is_fall else "NO_FALL"
            conf = score if is_fall else round(1.0 - score, 3)
            writer.writerow([db3_name, i, f"{ts:.3f}", pred_label, f"{conf:.3f}"])

    # Pass 3: Render Annotated Video
    if save_video:
        annotated_video_path = results_dir / f"{db3_stem}_annotated.mp4"
        cap = cv2.VideoCapture(str(preview_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (640, 480))

        render_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            is_fall = render_idx in frame_event_map
            ev_id = frame_event_map.get(render_idx, 0)
            f_data = frame_results[render_idx] if render_idx < len(frame_results) else {}

            annotated_frame = draw_annotation(
                frame,
                f_data,
                is_fall_frame=is_fall,
                event_id=ev_id,
                db3_name=db3_name,
                fps=fps,
                frame_idx=render_idx + 1,
                total_frames=total_frames,
            )
            out.write(annotated_frame)
            render_idx += 1

        cap.release()
        out.release()

    # Calculate summary metrics
    first_fall = f"{events[0]['start_time_seconds']:.2f}" if events else ""
    last_fall = f"{events[-1]['end_time_seconds']:.2f}" if events else ""
    max_conf = round(float(np.max(frame_scores)), 3) if frame_scores else 0.0

    summary_entry = {
        "source_db3": db3_name,
        "duration_seconds": round(duration_s, 2),
        "total_frames": total_frames,
        "fall_detected": fall_detected,
        "number_of_fall_events": num_events,
        "first_fall_time": first_fall,
        "last_fall_time": last_fall,
        "maximum_confidence": max_conf,
    }

    print(
        f"  --> Result: {fall_detected} | Events: {num_events} | Max Conf: {max_conf:.3f} "
        f"{f'| First Fall: {first_fall}s' if first_fall else ''}"
    )

    formatted_events = []
    for ev in events:
        formatted_events.append({
            "source_db3": db3_name,
            "event_id": ev["event_id"],
            "start_time_seconds": ev["start_time_seconds"],
            "end_time_seconds": ev["end_time_seconds"],
            "confidence": ev["confidence"],
        })

    return summary_entry, formatted_events


def main():
    parser = argparse.ArgumentParser(description="Batch Automatic Fall Detection Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/detector_config.yaml",
        help="Path to detector config file",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Run on single DB3 recording name or stem (e.g. 20260817_153540)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip generating annotated preview MP4 videos to speed up processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pseudo-labels and visualizations",
    )

    args = parser.parse_args()

    config_path = base_dir / args.config
    extracted_dir = base_dir / "extracted"
    pseudo_labels_dir = base_dir / "data" / "pseudo_labels"
    results_dir = base_dir / "results" / "automatic_detection"

    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = FallDetector(config_path)

    # Discover videos
    if args.file:
        stem = args.file.replace(".db3", "")
        target_dirs = [extracted_dir / stem]
    else:
        target_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir() and (d / "preview.mp4").exists()])

    print("=" * 80)
    print("      AUTOMATIC FALL DETECTION BASELINE PIPELINE")
    print("=" * 80)
    print(f"Total recordings to process: {len(target_dirs)}")
    print(f"Configuration: {config_path}")
    print(f"Pseudo-labels output: {pseudo_labels_dir}")
    print(f"Visualizations output: {results_dir}")
    print("=" * 80)

    all_summaries = []
    all_events = []

    # Read existing summary if any
    summary_csv_path = base_dir / "results" / "automatic_detection_summary.csv"
    events_csv_path = pseudo_labels_dir / "fall_events.csv"

    for idx, v_dir in enumerate(target_dirs, start=1):
        stem = v_dir.name
        print(f"\n[{idx}/{len(target_dirs)}] Processing: {stem} ...")
        summary_row, events = process_video(
            db3_stem=stem,
            detector=detector,
            extracted_dir=extracted_dir,
            pseudo_labels_dir=pseudo_labels_dir,
            results_dir=results_dir,
            save_video=not args.no_video,
        )
        if summary_row:
            all_summaries.append(summary_row)
            all_events.extend(events)

    # Write summary CSV
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_db3",
                "duration_seconds",
                "total_frames",
                "fall_detected",
                "number_of_fall_events",
                "first_fall_time",
                "last_fall_time",
                "maximum_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    # Write fall events CSV
    with open(events_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_db3",
                "event_id",
                "start_time_seconds",
                "end_time_seconds",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(all_events)

    # Consolidated frame predictions CSV
    consolidated_csv = pseudo_labels_dir / "all_frame_predictions.csv"
    with open(consolidated_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "source_db3",
            "frame_number",
            "timestamp_seconds",
            "prediction",
            "confidence",
        ])
        for v_dir in target_dirs:
            stem = v_dir.name
            fc_path = pseudo_labels_dir / f"{stem}_predictions.csv"
            if fc_path.exists():
                with open(fc_path, "r", encoding="utf-8") as f_in:
                    reader = csv.reader(f_in)
                    next(reader, None)  # Skip header
                    for row in reader:
                        writer.writerow(row)

    print("\n" + "=" * 80)
    print("                     BATCH RUN COMPLETE SUMMARY")
    print("=" * 80)
    total_recs = len(all_summaries)
    fall_recs = sum(1 for s in all_summaries if s["fall_detected"] == "FALL")
    nofall_recs = total_recs - fall_recs
    total_events = len(all_events)
    total_dur = sum(s["duration_seconds"] for s in all_summaries)

    print(f"Total Recordings Processed:     {total_recs}")
    print(f"Recordings with Fall Events:    {fall_recs} ({(fall_recs/total_recs)*100:.1f}%)")
    print(f"Recordings with Normal Activity:{nofall_recs} ({(nofall_recs/total_recs)*100:.1f}%)")
    print(f"Total Fall Events Segmented:    {total_events}")
    print(f"Total Analyzed Duration:        {total_dur:.2f}s ({total_dur/60:.2f} mins)")
    print(f"Summary CSV:                    {summary_csv_path}")
    print(f"Fall Events CSV:                {events_csv_path}")
    print(f"Annotated Videos Directory:     {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

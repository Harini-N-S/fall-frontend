#!/usr/bin/env python3
"""
run_refined_detection.py - Full Pipeline for Refined Dynamic Fall Detection.

Processes all 36 recordings with 3-state finite state machine (NORMAL -> FALLING -> LYING_DOWN),
generates annotated visualization videos, refined pseudo-labels, comparison metrics against baseline,
and updated quality report.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from fall_detector_refined import RefinedFallDetector, COCO_SKELETON


def draw_refined_annotation(
    frame: np.ndarray,
    record: dict,
    db3_name: str,
    fps: float,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    state = record.get("state", "NORMAL")
    event_id = record.get("event_id", 0)
    score = record.get("raw_score", 0.0)
    conf = record.get("confidence", 0.0)
    ts = record.get("timestamp_seconds", 0.0)
    box = record.get("box")
    kpts = record.get("keypoints")
    kpts_conf = record.get("keypoints_conf")
    ar = record.get("ar", 1.5)
    angle = record.get("angle_deg", 10.0)
    v_down = record.get("v_down", 0.0)

    # Color Palette:
    # FALLING -> Red
    # LYING_DOWN -> Orange/Amber
    # NORMAL -> Green
    if state == "FALLING":
        primary_color = (0, 0, 255)      # Red
        bg_color = (0, 0, 180)
        status_text = f"  FALL EVENT #{event_id}  "
    elif state == "LYING_DOWN":
        primary_color = (0, 140, 255)    # Amber / Orange
        bg_color = (0, 100, 200)
        status_text = "  POST-FALL LYING DOWN  "
    else:
        primary_color = (0, 220, 0)      # Green
        bg_color = (0, 140, 0)
        status_text = "  NORMAL ACTIVITY  "

    # 1. Draw Bounding Box
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), primary_color, 2)

        tag = f"{state} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), bg_color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 3, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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
    (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(out, (10, 8), (10 + stw, 40), bg_color, -1)
    cv2.putText(
        out,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Telemetry
    hud_info = (
        f"{db3_name} | T: {ts:5.2f}s | Frm: {frame_idx}/{total_frames} | "
        f"AR: {ar:.2f} | Ang: {angle:4.1f}° | V: {v_down:+.2f}"
    )
    cv2.putText(
        out,
        hud_info,
        (18 + stw, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Bottom watermark
    cv2.putText(
        out,
        "REFINED DETECTOR (NORMAL -> FALLING -> LYING_DOWN) - PSEUDO-LABEL",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def frame_generator(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def process_video_refined(
    db3_stem: str,
    detector: RefinedFallDetector,
    extracted_dir: Path,
    pseudo_labels_dir: Path,
    results_dir: Path,
    save_video: bool = True,
) -> Tuple[dict, List[dict]]:
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
            for row in csv.DictReader(f):
                timestamps_s.append(float(row["timestamp_seconds"]))

    cap = cv2.VideoCapture(str(preview_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if len(timestamps_s) != total_frames:
        timestamps_s = [i / fps for i in range(total_frames)]

    duration_s = timestamps_s[-1] if timestamps_s else 0.0

    print(f"Processing {db3_name}: {total_frames} frames ({duration_s:.2f}s @ {fps:.1f}fps)...")

    # Pass 1: Run State Machine & Kinematics
    frame_records, events = detector.process_video_sequence(
        frames_generator=frame_generator(preview_path),
        timestamps_s=timestamps_s,
        fps=fps,
    )

    num_events = len(events)
    fall_detected = "FALL" if num_events > 0 else "NO_FALL"

    fall_frames_cnt = sum(1 for r in frame_records if r["prediction"] == "FALL")
    lying_frames_cnt = sum(1 for r in frame_records if r["state"] == "LYING_DOWN")
    normal_frames_cnt = total_frames - fall_frames_cnt - lying_frames_cnt

    # Write frame-level pseudo-labels CSV
    frame_csv_path = pseudo_labels_dir / f"{db3_stem}_predictions.csv"
    with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_db3",
            "frame_number",
            "timestamp_seconds",
            "state",
            "prediction",
            "confidence",
        ])
        for r in frame_records:
            writer.writerow([
                db3_name,
                r["frame_number"],
                f"{r['timestamp_seconds']:.3f}",
                r["state"],
                r["prediction"],
                f"{r['confidence']:.3f}",
            ])

    # Pass 2: Render Annotated Video
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
            rec = frame_records[render_idx] if render_idx < len(frame_records) else {}
            annotated_frame = draw_refined_annotation(
                frame,
                record=rec,
                db3_name=db3_name,
                fps=fps,
                frame_idx=render_idx + 1,
                total_frames=total_frames,
            )
            out.write(annotated_frame)
            render_idx += 1

        cap.release()
        out.release()

    # Format events
    formatted_events = []
    for ev in events:
        formatted_events.append({
            "source_db3": db3_name,
            "event_id": ev["event_id"],
            "start_time_seconds": ev["start_time_seconds"],
            "end_time_seconds": ev["end_time_seconds"],
            "confidence": ev["confidence"],
        })

    first_fall = f"{events[0]['start_time_seconds']:.2f}" if events else ""
    last_fall = f"{events[-1]['end_time_seconds']:.2f}" if events else ""
    max_conf = max([ev["confidence"] for ev in events]) if events else (
        round(float(np.max([r['raw_score'] for r in frame_records])), 3) if frame_records else 0.0
    )

    summary_entry = {
        "source_db3": db3_name,
        "duration_seconds": round(duration_s, 2),
        "total_frames": total_frames,
        "fall_detected": fall_detected,
        "number_of_fall_events": num_events,
        "first_fall_time": first_fall,
        "last_fall_time": last_fall,
        "maximum_confidence": max_conf,
        "fall_frames": fall_frames_cnt,
        "lying_down_frames": lying_frames_cnt,
        "normal_frames": normal_frames_cnt,
    }

    event_str = ", ".join([f"#{ev['event_id']}: {ev['start_time_seconds']}s->{ev['end_time_seconds']}s" for ev in events]) if events else "None"
    print(f"  --> Result: {fall_detected} | Events: {num_events} ({event_str}) | FALL Frames: {fall_frames_cnt} ({fall_frames_cnt/total_frames*100:.1f}%) | LYING_DOWN: {lying_frames_cnt}")

    return summary_entry, formatted_events


def build_comparison(
    baseline_summary_path: Path,
    refined_summaries: List[Dict],
    comparison_csv_path: Path,
):
    """
    Generate side-by-side comparison.csv between baseline and refined detectors.
    """
    baseline_data = {}
    if baseline_summary_path.exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baseline_data[row["source_db3"]] = row

    baseline_events_path = baseline_summary_path.parent.parent / "data" / "pseudo_labels" / "fall_events.csv"
    baseline_events_by_db3 = {}
    if baseline_events_path.exists():
        with open(baseline_events_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                db3 = row["source_db3"]
                dur = float(row["end_time_seconds"]) - float(row["start_time_seconds"])
                baseline_events_by_db3[db3] = baseline_events_by_db3.get(db3, 0.0) + dur

    refined_events_by_db3 = {}
    refined_events_path = baseline_summary_path.parent.parent / "data" / "pseudo_labels_refined" / "fall_events.csv"
    if refined_events_path.exists():
        with open(refined_events_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                db3 = row["source_db3"]
                dur = float(row["end_time_seconds"]) - float(row["start_time_seconds"])
                refined_events_by_db3[db3] = refined_events_by_db3.get(db3, 0.0) + dur

    rows = []
    for r_entry in refined_summaries:
        db3 = r_entry["source_db3"]
        b_entry = baseline_data.get(db3, {})

        b_events = int(b_entry.get("number_of_fall_events", 0))
        r_events = r_entry["number_of_fall_events"]

        b_dur = round(baseline_events_by_db3.get(db3, 0.0), 2)
        r_dur = round(refined_events_by_db3.get(db3, 0.0), 2)

        tot_frames = r_entry["total_frames"] or 1
        r_fall_pct = round((r_entry["fall_frames"] / tot_frames) * 100, 1)

        # Baseline fall pct from frames
        b_frames_csv = baseline_summary_path.parent.parent / "data" / "pseudo_labels" / f"{db3.replace('.db3', '')}_predictions.csv"
        b_fall_cnt = 0
        if b_frames_csv.exists():
            with open(b_frames_csv, "r", encoding="utf-8") as fb:
                for row in csv.DictReader(fb):
                    if row["prediction"] == "FALL":
                        b_fall_cnt += 1
        b_fall_pct = round((b_fall_cnt / tot_frames) * 100, 1)

        rows.append({
            "source_db3": db3,
            "baseline_events": b_events,
            "refined_events": r_events,
            "baseline_fall_duration": f"{b_dur:.2f}s",
            "refined_fall_duration": f"{r_dur:.2f}s",
            "baseline_fall_percentage": f"{b_fall_pct:.1f}%",
            "refined_fall_percentage": f"{r_fall_pct:.1f}%",
        })

    with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_db3",
                "baseline_events",
                "refined_events",
                "baseline_fall_duration",
                "refined_fall_duration",
                "baseline_fall_percentage",
                "refined_fall_percentage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_quality_report(
    refined_events: List[Dict],
    summary_entries: List[Dict],
    quality_report_csv: Path,
):
    video_summary = {s["source_db3"]: s for s in summary_entries}

    report_rows = []
    for ev in refined_events:
        db3 = ev["source_db3"]
        v_info = video_summary.get(db3, {})
        v_dur = v_info.get("duration_seconds", 0.0)
        tot_f = v_info.get("total_frames", 1)
        fall_f = v_info.get("fall_frames", 0)
        v_events_cnt = v_info.get("number_of_fall_events", 1)

        st = float(ev["start_time_seconds"])
        et = float(ev["end_time_seconds"])
        dur = round(et - st, 2)
        conf = float(ev["confidence"])
        pct = round((fall_f / tot_f) * 100, 1)

        flags = []
        if st <= 0.05:
            flags.append("STARTS_AT_ZERO")
        if et >= v_dur - 0.5 and v_dur > 3.0:
            flags.append("EXTENDS_TO_END")
        if dur > 8.0:
            flags.append("PROLONGED_DURATION(>8s)")
        elif dur > 4.0:
            flags.append("LONG_DURATION(>4s)")
        if pct > 50.0:
            flags.append("HIGH_FALL_PCT(>50%)")
        if v_events_cnt >= 4:
            flags.append(f"MANY_EVENTS_IN_VIDEO({v_events_cnt})")

        flag_str = "; ".join(flags) if flags else "OK (Plausible Dynamic Fall)"

        report_rows.append({
            "source_db3": db3,
            "event_id": ev["event_id"],
            "start_time": f"{st:.2f}",
            "end_time": f"{et:.2f}",
            "duration": f"{dur:.2f}",
            "confidence": f"{conf:.3f}",
            "fall_frame_count": fall_f,
            "video_fall_percentage": f"{pct:.1f}%",
            "quality_flag": flag_str,
        })

    with open(quality_report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_db3",
                "event_id",
                "start_time",
                "end_time",
                "duration",
                "confidence",
                "fall_frame_count",
                "video_fall_percentage",
                "quality_flag",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)


def main():
    parser = argparse.ArgumentParser(description="Batch Refined Fall Detection Runner")
    parser.add_argument("--config", type=str, default="config/detector_config.yaml")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--no-video", action="store_true")

    args = parser.parse_args()

    config_path = base_dir / args.config
    extracted_dir = base_dir / "extracted"
    pseudo_labels_dir = base_dir / "data" / "pseudo_labels_refined"
    results_dir = base_dir / "results" / "refined_detection"

    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = RefinedFallDetector(config_path)

    if args.file:
        stem = args.file.replace(".db3", "")
        target_dirs = [extracted_dir / stem]
    else:
        target_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir() and (d / "preview.mp4").exists()])

    print("=" * 80)
    print("      REFINED DYNAMIC FALL DETECTION BATCH PIPELINE")
    print("=" * 80)
    print(f"Total recordings to process: {len(target_dirs)}")
    print(f"Configuration: {config_path}")
    print(f"Pseudo-labels output: {pseudo_labels_dir}")
    print(f"Results output: {results_dir}")
    print("=" * 80)

    all_summaries = []
    all_events = []

    summary_csv_path = results_dir / "refined_detection_summary.csv"
    events_csv_path = pseudo_labels_dir / "fall_events.csv"
    comparison_csv_path = results_dir / "comparison.csv"
    quality_report_csv_path = results_dir / "pseudo_label_quality_report.csv"
    baseline_summary_path = base_dir / "results" / "automatic_detection_summary.csv"

    for idx, v_dir in enumerate(target_dirs, start=1):
        stem = v_dir.name
        print(f"\n[{idx}/{len(target_dirs)}] Analyzing: {stem} ...")
        summary_row, events = process_video_refined(
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
                "fall_frames",
                "lying_down_frames",
                "normal_frames",
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
            "state",
            "prediction",
            "confidence",
        ])
        for v_dir in target_dirs:
            stem = v_dir.name
            fc_path = pseudo_labels_dir / f"{stem}_predictions.csv"
            if fc_path.exists():
                with open(fc_path, "r", encoding="utf-8") as f_in:
                    reader = csv.reader(f_in)
                    next(reader, None)
                    for row in reader:
                        writer.writerow(row)

    # Build Comparison CSV
    build_comparison(baseline_summary_path, all_summaries, comparison_csv_path)

    # Build Quality Report
    build_quality_report(all_events, all_summaries, quality_report_csv_path)

    print("\n" + "=" * 80)
    print("                     REFINED BATCH RUN COMPLETE")
    print("=" * 80)
    total_recs = len(all_summaries)
    fall_recs = sum(1 for s in all_summaries if s["fall_detected"] == "FALL")
    nofall_recs = total_recs - fall_recs
    total_events = len(all_events)

    tot_fall_frames = sum(s["fall_frames"] for s in all_summaries)
    tot_lying_frames = sum(s["lying_down_frames"] for s in all_summaries)
    tot_all_frames = sum(s["total_frames"] for s in all_summaries)

    print(f"Total Recordings Processed:     {total_recs}")
    print(f"Recordings with Fall Events:    {fall_recs} ({(fall_recs/total_recs)*100:.1f}%)")
    print(f"Recordings with Normal/Lying:   {nofall_recs} ({(nofall_recs/total_recs)*100:.1f}%)")
    print(f"Total Refined Fall Events:      {total_events}")
    print(f"Total FALL Frames:              {tot_fall_frames} ({(tot_fall_frames/tot_all_frames)*100:.1f}%)")
    print(f"Total LYING_DOWN Frames:        {tot_lying_frames} ({(tot_lying_frames/tot_all_frames)*100:.1f}%)")
    print(f"Summary CSV:                    {summary_csv_path}")
    print(f"Comparison CSV:                 {comparison_csv_path}")
    print(f"Quality Report CSV:             {quality_report_csv_path}")
    print(f"Annotated Videos Directory:     {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

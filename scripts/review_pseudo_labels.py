#!/usr/bin/env python3
"""
review_pseudo_labels.py - Spot-Check & Validation Mechanism for Fall Pseudo-Labels.

Allows reviewing detected fall events (just the short event intervals, rather than
watching all 22,773 frames) to promote verified events to ground-truth labels.
"""

import sys
import csv
from pathlib import Path
import cv2

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from label_manager import save_video_label, save_fall_events_for_db3


def review_events():
    pseudo_events_csv = base_dir / "data" / "pseudo_labels" / "fall_events.csv"
    summary_csv = base_dir / "results" / "automatic_detection_summary.csv"
    results_video_dir = base_dir / "results" / "automatic_detection"

    if not pseudo_events_csv.exists():
        print("[ERROR] Pseudo-labels fall_events.csv not found. Run automatic detection first.")
        return

    events = []
    with open(pseudo_events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)

    print("=" * 80)
    print("       FALL DETECTION PSEUDO-LABEL EVENT REVIEWER")
    print("=" * 80)
    print(f"Total Detected Fall Events to Review: {len(events)}")
    print("Review only the flagged fall intervals to quickly verify or reject predictions.")
    print("=" * 80)

    for idx, ev in enumerate(events, start=1):
        db3_name = ev["source_db3"]
        stem = db3_name.replace(".db3", "")
        annotated_video = results_video_dir / f"{stem}_annotated.mp4"
        start_s = float(ev["start_time_seconds"])
        end_s = float(ev["end_time_seconds"])
        conf = float(ev["confidence"])

        print(f"\n[{idx}/{len(events)}] {db3_name} | Event #{ev['event_id']}")
        print(f"  Detected Interval: {start_s:.2f}s -> {end_s:.2f}s (Duration: {end_s - start_s:.2f}s)")
        print(f"  Model Confidence:  {conf:.3f}")

        if not annotated_video.exists():
            print(f"  [WARN] Annotated video missing: {annotated_video.name}")
            continue

        choice = input("  Play annotated clip? [Y/n/q]: ").strip().lower()
        if choice == "q":
            break
        elif choice != "n":
            # Play interval with 1.5s margin
            cap = cv2.VideoCapture(str(annotated_video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            start_frame = max(0, int((start_s - 1.5) * fps))
            end_frame = int((end_s + 2.0) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            win_name = f"Reviewing: {db3_name} Event #{ev['event_id']} (Q: Close clip)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 720, 540)

            while cap.isOpened():
                curr_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if curr_f > end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(win_name, frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF in (27, ord('q'), ord('Q')):
                    break
            cap.release()
            cv2.destroyWindow(win_name)

        print("  Verdict:")
        print("    [1] CONFIRM as TRUE FALL (promote to ground-truth)")
        print("    [2] REJECT as FALSE POSITIVE (normal activity / bending)")
        print("    [s] Skip / leave as unverified pseudo-label")

        verdict = input("  Selection (1/2/s): ").strip()
        if verdict == "1":
            notes = input("  Notes (optional): ").strip()
            save_video_label(db3_name, "FALL", notes, base_dir)
            save_fall_events_for_db3(
                db3_name,
                [{
                    "event_id": ev["event_id"],
                    "start_time_seconds": start_s,
                    "end_time_seconds": end_s,
                    "notes": f"VERIFIED: {notes}",
                }],
                base_dir,
            )
            print(f"  --> Confirmed & saved as verified GROUND TRUTH in data/labels/")
        elif verdict == "2":
            notes = input("  Notes (optional): ").strip()
            save_video_label(db3_name, "NO_FALL", f"REJECTED PSEUDO-LABEL: {notes}", base_dir)
            save_fall_events_for_db3(db3_name, [], base_dir)
            print(f"  --> Marked as NO_FALL in data/labels/")
        else:
            print("  --> Kept as unverified pseudo-label.")

    print("\nReview session complete.")


if __name__ == "__main__":
    review_events()

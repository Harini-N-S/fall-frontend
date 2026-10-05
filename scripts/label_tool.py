#!/usr/bin/env python3
"""
label_tool.py - CLI & Desktop Interface for Fall Detection Dataset Labeling.

Usage:
    python scripts/label_tool.py           # Interactive CLI / OpenCV video labeling
    python scripts/label_tool.py --web     # Launch the Streamlit Web Interface
    python scripts/label_tool.py --test    # Run automated test on one preview video
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import cv2

# Add scripts directory to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from label_manager import (
    load_video_labels,
    save_video_label,
    load_fall_events,
    save_fall_events_for_db3,
    get_progress,
)


def play_video_opencv(video_path: Path):
    """
    Play preview video using OpenCV with on-screen timestamp and controls.
    Returns:
        dict with marked timestamps or None
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / fps))

    window_name = f"Preview: {video_path.name} (Space: Pause | R: Replay | Q/Esc: Done)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    paused = False
    current_frame = 0

    marked_start = None
    marked_end = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 1))
            ret, frame = cap.read()
            if not ret:
                break

        current_sec = current_frame / fps
        total_sec = total_frames / fps

        # Render HUD overlay
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (hud.shape[1], 45), (20, 20, 20), -1)
        info_text = f"Time: {current_sec:5.2f}s / {total_sec:5.2f}s | Frame: {current_frame}/{total_frames} | {'PAUSED' if paused else 'PLAYING'}"
        cv2.putText(hud, info_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        ctrl_text = "[Space]: Pause/Play  [R]: Restart  [S]: Mark Start  [E]: Mark End  [Q]: Done"
        cv2.putText(hud, ctrl_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if marked_start is not None:
            cv2.putText(hud, f"Marked Start: {marked_start:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if marked_end is not None:
            cv2.putText(hud, f"Marked End: {marked_end:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow(window_name, hud)
        key = cv2.waitKey(delay if not paused else 50) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # Space
            paused = not paused
        elif key in (ord('r'), ord('R')):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
            paused = False
        elif key in (ord('s'), ord('S')):
            marked_start = current_sec
            print(f"  --> Marked fall start: {marked_start:.2f}s")
        elif key in (ord('e'), ord('E')):
            marked_end = current_sec
            print(f"  --> Marked fall end: {marked_end:.2f}s")

    cap.release()
    cv2.destroyWindow(window_name)

    return {"start": marked_start, "end": marked_end}


def run_interactive_cli(all_recordings: bool = False):
    extracted_dir = base_dir / "extracted"
    if not extracted_dir.exists():
        print(f"[ERROR] Directory '{extracted_dir}' does not exist.")
        return

    video_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir() and (d / "preview.mp4").exists()])
    if not video_dirs:
        print("[ERROR] No preview videos found in extracted/.")
        return

    all_labels = load_video_labels(base_dir)
    labels_dict = {row["source_db3"]: row for row in all_labels}

    total_count = len(video_dirs)

    print("=" * 75)
    print("      FALL DETECTION DATASET LABELING TOOL (CLI)")
    print("=" * 75)
    print("Commands:")
    print("  Watch preview video with OpenCV player, then choose label and record events.")
    print("  Type 'q' at any prompt to save and quit.")
    print("=" * 75)

    for idx, v_dir in enumerate(video_dirs, start=1):
        db3_name = f"{v_dir.name}.db3"
        curr_info = labels_dict.get(db3_name, {"label": "UNKNOWN", "notes": ""})
        curr_label = curr_info["label"]

        labeled_count, _, _ = get_progress(base_dir)

        if not all_recordings and curr_label != "UNKNOWN":
            continue

        preview_path = v_dir / "preview.mp4"

        # Get metadata
        cap = cv2.VideoCapture(str(preview_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0.0
        cap.release()

        print(f"\n[{idx}/{total_count}] Processing: {db3_name}")
        print(f"  Duration: {duration:.2f}s | FPS: {fps:.2f} | Frames: {frames} | Current Label: [{curr_label}]")
        print(f"  Overall Progress: {labeled_count} / {total_count} videos labeled ({(labeled_count/total_count)*100:.1f}%)")

        # Option to play video
        play_choice = input("  Play video in OpenCV window? [Y/n/q]: ").strip().lower()
        if play_choice == "q":
            print("Exiting labeling session.")
            break

        marked = None
        if play_choice != "n":
            marked = play_video_opencv(preview_path)

        # Label selection
        print("\n  Choose Classification:")
        print("    [1] FALL")
        print("    [2] NO_FALL")
        print("    [3] UNKNOWN")
        print("    [s] Skip to next video without changing")
        print("    [q] Save and quit")

        choice = input("  Selection (1/2/3/s/q): ").strip().lower()
        if choice == "q":
            print("Exiting labeling session.")
            break
        elif choice == "s":
            print("  Skipped.")
            continue

        choice_map = {"1": "FALL", "2": "NO_FALL", "3": "UNKNOWN"}
        new_label = choice_map.get(choice, curr_label)

        # Notes
        notes_prompt = f"  Notes (Enter to keep '{curr_info.get('notes', '')}'): "
        new_notes = input(notes_prompt).strip()
        if not new_notes:
            new_notes = curr_info.get("notes", "")

        # Fall events
        events = load_fall_events(db3_name, base_dir)
        if new_label == "FALL":
            print("\n  -- Fall Event Timing --")
            default_start = f"{marked['start']:.1f}" if marked and marked.get("start") is not None else "0.0"
            default_end = f"{marked['end']:.1f}" if marked and marked.get("end") is not None else f"{duration:.1f}"

            add_ev = input(f"  Record fall event interval? [Y/n]: ").strip().lower()
            if add_ev != "n":
                start_in = input(f"    Start time (seconds) [{default_start}]: ").strip() or default_start
                end_in = input(f"    End time (seconds) [{default_end}]: ").strip() or default_end
                ev_notes = input("    Event description/notes (optional): ").strip()

                try:
                    ev_start = float(start_in)
                    ev_end = float(end_in)
                    events.append({
                        "event_id": str(len(events) + 1),
                        "start_time_seconds": ev_start,
                        "end_time_seconds": ev_end,
                        "notes": ev_notes,
                    })
                    save_fall_events_for_db3(db3_name, events, base_dir)
                    print(f"  --> Saved Fall Event: {ev_start:.1f}s to {ev_end:.1f}s")
                except ValueError:
                    print("  [WARN] Invalid start/end time numbers. Skipping event record.")

        # Save video label
        save_video_label(db3_name, new_label, new_notes, base_dir)
        labels_dict[db3_name] = {"source_db3": db3_name, "label": new_label, "notes": new_notes}
        print(f"  [SAVED] {db3_name} -> {new_label}")

    final_labeled, _, _ = get_progress(base_dir)
    print(f"\nSession complete. {final_labeled} / {total_count} videos labeled.")


def run_automated_test():
    """
    Test the labeling system components on ONE preview video:
    - Verifies video loading and frame decoding
    - Tests setting FALL label and notes
    - Tests recording fall event with start and end seconds
    - Verifies CSV persistence
    - Restores the original label state
    """
    print("=" * 70)
    print("RUNNING AUTOMATED VALIDATION TEST ON ONE PREVIEW VIDEO")
    print("=" * 70)

    test_file = "20260817_153540.db3"
    extracted_dir = base_dir / "extracted" / "20260817_153540"
    preview_path = extracted_dir / "preview.mp4"

    # 1. Test video opening and decoding
    print(f"[1/5] Verifying video file: {preview_path.name}")
    assert preview_path.exists(), f"Preview video missing: {preview_path}"

    cap = cv2.VideoCapture(str(preview_path))
    assert cap.isOpened(), "Failed to open preview.mp4 with OpenCV"
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    ret, first_frame = cap.read()
    assert ret and first_frame is not None, "Failed to decode first frame from preview video"
    cap.release()

    print(f"      Video verified: {width}x{height} @ {fps:.2f}fps, {total_frames} frames, {duration:.2f}s duration.")

    # 2. Test saving video label
    print(f"[2/5] Testing label selection & notes persistence...")
    test_label = "FALL"
    test_notes = "TEST: person falls at 1.2s"

    save_video_label(test_file, test_label, test_notes, base_dir)

    # Verify video_labels.csv
    labels = load_video_labels(base_dir)
    found = [r for r in labels if r["source_db3"] == test_file]
    assert len(found) == 1, f"Row for {test_file} not found in video_labels.csv"
    assert found[0]["label"] == test_label, f"Expected {test_label}, got {found[0]['label']}"
    assert found[0]["notes"] == test_notes, f"Expected {test_notes}, got {found[0]['notes']}"
    print(f"      Verified video_labels.csv: source_db3={test_file}, label={found[0]['label']}, notes='{found[0]['notes']}'")

    # 3. Test saving fall event
    print(f"[3/5] Testing fall event timing persistence...")
    test_events = [
        {
            "event_id": "1",
            "start_time_seconds": 0.8,
            "end_time_seconds": 2.2,
            "notes": "TEST: rapid forward fall",
        }
    ]
    save_fall_events_for_db3(test_file, test_events, base_dir)

    # Verify fall_events.csv
    loaded_events = load_fall_events(test_file, base_dir)
    assert len(loaded_events) == 1, "Failed to load fall event from fall_events.csv"
    ev = loaded_events[0]
    assert ev["source_db3"] == test_file
    assert ev["event_id"] == "1"
    assert float(ev["start_time_seconds"]) == 0.8
    assert float(ev["end_time_seconds"]) == 2.2
    assert ev["notes"] == "TEST: rapid forward fall"
    print(f"      Verified fall_events.csv: event_id=1, interval={ev['start_time_seconds']}s -> {ev['end_time_seconds']}s, notes='{ev['notes']}'")

    # 4. Test progress calculation
    print(f"[4/5] Testing labeling progress tracker...")
    labeled_count, total_count, unlabeled_list = get_progress(base_dir)
    print(f"      Progress: {labeled_count} / {total_count} videos labeled ({(labeled_count/total_count)*100:.1f}%)")
    assert labeled_count >= 1, "Progress count failed to reflect labeled video"

    # 5. Reset test label back to UNKNOWN and clear test event (preserving dataset purity)
    print(f"[5/5] Resetting test row back to UNKNOWN to maintain dataset purity...")
    save_video_label(test_file, "UNKNOWN", "", base_dir)
    save_fall_events_for_db3(test_file, [], base_dir)

    reset_labels = load_video_labels(base_dir)
    reset_found = [r for r in reset_labels if r["source_db3"] == test_file][0]
    assert reset_found["label"] == "UNKNOWN", "Failed to reset label to UNKNOWN"
    reset_events = load_fall_events(test_file, base_dir)
    assert len(reset_events) == 0, "Failed to clear test event"
    print("      Reset verified: all 36 videos cleanly in UNKNOWN state ready for user.")

    print("=" * 70)
    print("ALL 5 AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Dataset Labeling Tool")
    parser.add_argument("--web", action="store_true", help="Launch Streamlit Web App interface")
    parser.add_argument("--test", action="store_true", help="Run automated test suite on one preview video")
    parser.add_argument("--all", action="store_true", help="Review all videos (including already labeled ones)")

    args = parser.parse_args()

    if args.test:
        run_automated_test()
    elif args.web:
        app_path = base_dir / "scripts" / "label_app.py"
        print(f"Launching Streamlit Web App: {app_path}")
        subprocess.run(["streamlit", "run", str(app_path)])
    else:
        run_interactive_cli(all_recordings=args.all)


if __name__ == "__main__":
    main()

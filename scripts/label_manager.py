"""
label_manager.py - Data access and management layer for video labels and fall events.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_paths(base_dir: Optional[Path] = None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    extracted_dir = base_dir / "extracted"
    labels_dir = base_dir / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    video_labels_csv = labels_dir / "video_labels.csv"
    fall_events_csv = labels_dir / "fall_events.csv"
    return base_dir, extracted_dir, labels_dir, video_labels_csv, fall_events_csv


def init_labels_if_needed(base_dir: Optional[Path] = None):
    _, extracted_dir, _, video_labels_csv, fall_events_csv = get_paths(base_dir)

    # Discover all DB3 folders with preview.mp4
    video_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir() and (d / "preview.mp4").exists()])

    if not video_labels_csv.exists():
        with open(video_labels_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_db3", "label", "notes"])
            for d in video_dirs:
                writer.writerow([f"{d.name}.db3", "UNKNOWN", ""])

    if not fall_events_csv.exists():
        with open(fall_events_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_db3", "event_id", "start_time_seconds", "end_time_seconds", "notes"])


def load_video_labels(base_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    _, _, _, video_labels_csv, _ = get_paths(base_dir)
    init_labels_if_needed(base_dir)
    labels = []
    with open(video_labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row)
    return labels


def save_video_label(source_db3: str, label: str, notes: str = "", base_dir: Optional[Path] = None):
    _, _, _, video_labels_csv, _ = get_paths(base_dir)
    init_labels_if_needed(base_dir)

    rows = []
    updated = False
    with open(video_labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["source_db3", "label", "notes"]
        for row in reader:
            if row["source_db3"] == source_db3:
                row["label"] = label
                row["notes"] = notes
                updated = True
            rows.append(row)

    if not updated:
        rows.append({"source_db3": source_db3, "label": label, "notes": notes})

    with open(video_labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_fall_events(source_db3: Optional[str] = None, base_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    _, _, _, _, fall_events_csv = get_paths(base_dir)
    init_labels_if_needed(base_dir)
    events = []
    with open(fall_events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if source_db3 is None or row["source_db3"] == source_db3:
                events.append(row)
    return events


def save_fall_events_for_db3(source_db3: str, events: List[Dict[str, str]], base_dir: Optional[Path] = None):
    """
    Replace all fall event rows for a given source_db3 with the provided list of events.
    """
    _, _, _, _, fall_events_csv = get_paths(base_dir)
    init_labels_if_needed(base_dir)

    all_events = []
    with open(fall_events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["source_db3", "event_id", "start_time_seconds", "end_time_seconds", "notes"]
        for row in reader:
            if row["source_db3"] != source_db3:
                all_events.append(row)

    for idx, ev in enumerate(events, start=1):
        all_events.append({
            "source_db3": source_db3,
            "event_id": str(ev.get("event_id", idx)),
            "start_time_seconds": f"{float(ev['start_time_seconds']):.1f}",
            "end_time_seconds": f"{float(ev['end_time_seconds']):.1f}",
            "notes": ev.get("notes", ""),
        })

    with open(fall_events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)


def get_progress(base_dir: Optional[Path] = None) -> Tuple[int, int, List[str]]:
    """
    Returns:
        (labeled_count, total_count, list_of_unlabeled_db3_names)
    """
    labels = load_video_labels(base_dir)
    total = len(labels)
    labeled = sum(1 for row in labels if row["label"] in ("FALL", "NO_FALL"))
    unlabeled = [row["source_db3"] for row in labels if row["label"] == "UNKNOWN"]
    return labeled, total, unlabeled

"""
label_app.py - Interactive Streamlit Web Interface for Fall Detection Dataset Labeling.

Launch with:
    streamlit run scripts/label_app.py
"""

import sys
from pathlib import Path
import streamlit as st
import cv2

# Add parent directory to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from label_manager import (
    load_video_labels,
    save_video_label,
    load_fall_events,
    save_fall_events_for_db3,
    get_progress,
)

st.set_page_config(
    page_title="Fall Detection Dataset Labeler",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Fall Detection Dataset Labeling Tool")

# Check directories
extracted_dir = base_dir / "extracted"
if not extracted_dir.exists():
    st.error("Extracted directory not found! Run `python scripts/extract_frames.py` first.")
    st.stop()

# Load all video directories
video_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir() and (d / "preview.mp4").exists()])
if not video_dirs:
    st.error("No extracted videos with preview.mp4 found in `extracted/`.")
    st.stop()

# Get progress
all_labels = load_video_labels(base_dir)
labels_dict = {row["source_db3"]: row for row in all_labels}
labeled_count, total_count, unlabeled_list = get_progress(base_dir)

# Progress header
col_prog1, col_prog2, col_prog3 = st.columns([2, 1, 1])
with col_prog1:
    prog_ratio = labeled_count / max(1, total_count)
    st.progress(prog_ratio)
with col_prog2:
    st.metric("Progress", f"{labeled_count} / {total_count} labeled", f"{prog_ratio * 100:.1f}%")
with col_prog3:
    filter_mode = st.radio("Display:", ["All Recordings", "Unlabeled (UNKNOWN) Only"], horizontal=True)

# Filter candidate list
if filter_mode == "Unlabeled (UNKNOWN) Only":
    active_dirs = [d for d in video_dirs if labels_dict.get(f"{d.name}.db3", {}).get("label") == "UNKNOWN"]
    if not active_dirs:
        st.success("🎉 All recordings have been labeled! Switch to 'All Recordings' to review.")
        active_dirs = video_dirs
else:
    active_dirs = video_dirs

active_names = [f"{d.name}.db3" for d in active_dirs]

# Session state index
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# Bound index
st.session_state.selected_idx = max(0, min(st.session_state.selected_idx, len(active_names) - 1))

# Selection dropdown
selected_db3 = st.selectbox(
    "Select Recording:",
    active_names,
    index=st.session_state.selected_idx,
    key="db3_selector",
)
current_idx = active_names.index(selected_db3)
st.session_state.selected_idx = current_idx

current_dir_name = selected_db3.replace(".db3", "")
current_dir = extracted_dir / current_dir_name
preview_video_path = current_dir / "preview.mp4"

# Inspect video metadata
cap = cv2.VideoCapture(str(preview_video_path))
v_width, v_height, v_fps, v_count, v_duration = 640, 480, 30.0, 0, 0.0
if cap.isOpened():
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    v_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_duration = v_count / v_fps if v_fps > 0 else 0.0
    cap.release()

# Current label status
current_info = labels_dict.get(selected_db3, {"label": "UNKNOWN", "notes": ""})
current_events = load_fall_events(selected_db3, base_dir)

# Main layout: Video on Left, Annotation controls on Right
left_col, right_col = st.columns([1.2, 1.0], gap="medium")

with left_col:
    st.subheader(f"📹 {selected_db3}")
    st.caption(f"Resolution: {v_width}×{v_height} | FPS: {v_fps:.2f} | Frames: {v_count} | Duration: {v_duration:.2f}s")
    if preview_video_path.exists():
        st.video(str(preview_video_path))
    else:
        st.warning(f"Preview video not found at {preview_video_path}")

with right_col:
    st.subheader("📝 Annotation & Classification")
    
    # 1. Classification
    label_options = ["FALL", "NO_FALL", "UNKNOWN"]
    default_label_idx = label_options.index(current_info["label"]) if current_info["label"] in label_options else 2
    
    selected_label = st.radio(
        "Recording Classification:",
        label_options,
        index=default_label_idx,
        help="Select FALL if the recording contains one or more falls. Select NO_FALL for normal activities.",
        horizontal=True,
    )
    
    video_notes = st.text_area(
        "Recording Notes (optional):",
        value=current_info.get("notes", ""),
        placeholder="e.g., person walks, trips over stool, falls forward",
        height=70,
    )
    
    st.markdown("---")
    
    # 2. Fall Events Section
    st.subheader("⏱️ Fall Events Timeline")
    st.caption("Mark exact intervals where fall events occur within this recording.")
    
    # Display existing events
    if current_events:
        st.write("**Recorded Fall Events:**")
        for idx, ev in enumerate(current_events, start=1):
            st.info(f"**Event #{ev['event_id']}**: {ev['start_time_seconds']}s ➔ {ev['end_time_seconds']}s | *Notes*: {ev.get('notes', 'None')}")
            
    # Add new event inputs
    with st.expander("➕ Add / Record Fall Event", expanded=(selected_label == "FALL" and len(current_events) == 0)):
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            start_s = st.number_input(
                "Start Time (seconds):",
                min_value=0.0,
                max_value=max(0.1, float(v_duration)),
                value=0.0,
                step=0.1,
                format="%.1f",
            )
        with ev_col2:
            end_s = st.number_input(
                "End Time (seconds):",
                min_value=0.0,
                max_value=max(0.1, float(v_duration)),
                value=min(float(v_duration), start_s + 2.0),
                step=0.1,
                format="%.1f",
            )
        event_notes = st.text_input("Event Details / Notes:", placeholder="e.g., person stumbles and falls sideways")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Add This Event", type="secondary"):
                if end_s <= start_s:
                    st.error("End time must be greater than start time!")
                else:
                    new_ev = {
                        "event_id": str(len(current_events) + 1),
                        "start_time_seconds": start_s,
                        "end_time_seconds": end_s,
                        "notes": event_notes,
                    }
                    current_events.append(new_ev)
                    save_fall_events_for_db3(selected_db3, current_events, base_dir)
                    st.success(f"Added Event #{new_ev['event_id']} ({start_s}s - {end_s}s)")
                    st.rerun()
        with btn_col2:
            if current_events and st.button("Clear All Events For This Video", type="secondary"):
                save_fall_events_for_db3(selected_db3, [], base_dir)
                st.warning("Cleared fall events for this video.")
                st.rerun()

    st.markdown("---")
    
    # Save & Navigation buttons
    save_col1, save_col2, save_col3 = st.columns([1, 1.5, 1])
    
    with save_col1:
        if st.button("⬅️ Previous", disabled=(current_idx == 0)):
            st.session_state.selected_idx = max(0, current_idx - 1)
            st.rerun()
            
    with save_col2:
        if st.button("💾 Save & Next Recording ➔", type="primary", use_container_width=True):
            # Save label immediately
            save_video_label(selected_db3, selected_label, video_notes, base_dir)
            st.toast(f"Saved {selected_db3} as {selected_label}!", icon="✅")
            # Advance to next video
            if current_idx < len(active_names) - 1:
                st.session_state.selected_idx = current_idx + 1
            st.rerun()
            
    with save_col3:
        if st.button("Next ➡️", disabled=(current_idx == len(active_names) - 1)):
            st.session_state.selected_idx = min(len(active_names) - 1, current_idx + 1)
            st.rerun()

# Sidebar summary
with st.sidebar:
    st.subheader("📊 Dataset Label Overview")
    fall_cnt = sum(1 for row in all_labels if row["label"] == "FALL")
    nofall_cnt = sum(1 for row in all_labels if row["label"] == "NO_FALL")
    unknown_cnt = sum(1 for row in all_labels if row["label"] == "UNKNOWN")
    
    st.write(f"- **FALL**: {fall_cnt}")
    st.write(f"- **NO_FALL**: {nofall_cnt}")
    st.write(f"- **UNKNOWN**: {unknown_cnt}")
    
    st.markdown("---")
    st.caption("Files automatically saved to:")
    st.code("data/labels/video_labels.csv\ndata/labels/fall_events.csv")

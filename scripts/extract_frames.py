#!/usr/bin/env python3
"""
extract_frames.py - ROS 2 Bag (.db3) RGB Frame Extraction Pipeline

Extracts RGB frames from ROS 2 bag SQLite3 files containing Zstandard-compressed
sensor_msgs/msg/Image CDR payloads, saving them as sequential JPG images, generating
timestamps.csv, and optionally producing a preview MP4 video.
"""

import os
import sys
import argparse
import sqlite3
import struct
import csv
from pathlib import Path
import numpy as np
import pyarrow as pa
import cv2


def parse_ros2_image_cdr(raw_bytes: bytes):
    """
    Parse a ROS 2 CDR-encoded sensor_msgs/msg/Image message.

    Returns:
        tuple: (sec, nanosec, frame_id, height, width, encoding, is_bigendian, step, image_numpy)
    """
    # Offset 0..4: CDR Representation Header (usually 0x00, 0x01, 0x00, 0x00)
    offset = 4

    # std_msgs/Header:
    # builtin_interfaces/Time: int32 sec (4 bytes), uint32 nanosec (4 bytes)
    sec, nanosec = struct.unpack_from("<iI", raw_bytes, offset)
    offset += 8

    # string frame_id: uint32 length + characters
    frame_id_len = struct.unpack_from("<I", raw_bytes, offset)[0]
    offset += 4
    frame_id = (
        raw_bytes[offset : offset + frame_id_len]
        .decode("utf-8", errors="ignore")
        .rstrip("\x00")
    )
    offset += frame_id_len

    # Align to 4-byte boundary for next fields
    offset = (offset + 3) & ~3

    # uint32 height, uint32 width
    height, width = struct.unpack_from("<II", raw_bytes, offset)
    offset += 8

    # string encoding: uint32 length + characters
    encoding_len = struct.unpack_from("<I", raw_bytes, offset)[0]
    offset += 4
    encoding = (
        raw_bytes[offset : offset + encoding_len]
        .decode("utf-8", errors="ignore")
        .rstrip("\x00")
    )
    offset += encoding_len

    # uint8 is_bigendian (1 byte)
    is_bigendian = raw_bytes[offset]
    offset += 1

    # uint32 step (aligned to 4 bytes)
    offset = (offset + 3) & ~3
    step = struct.unpack_from("<I", raw_bytes, offset)[0]
    offset += 4

    # uint32 data length (aligned to 4 bytes)
    data_len = struct.unpack_from("<I", raw_bytes, offset)[0]
    offset += 4

    img_data = raw_bytes[offset : offset + data_len]

    # Handle standard RGB8 encoding
    if encoding in ("rgb8", "bgr8"):
        img_np = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding in ("mono8", "8UC1"):
        img_np = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
    elif encoding in ("mono16", "16UC1"):
        img_np = np.frombuffer(img_data, dtype=np.uint16).reshape((height, width))
    else:
        # Generic fallback
        img_np = np.frombuffer(img_data, dtype=np.uint8)

    return sec, nanosec, frame_id, height, width, encoding, is_bigendian, step, img_np


def create_preview_video(frames_dir: Path, output_video_path: Path, fps: float, width: int = 640, height: int = 480):
    """
    Create a preview MP4 video from sequential extracted JPG frames.
    """
    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    if not frame_files:
        print(f"  [WARN] No frames found in {frames_dir} for preview video.")
        return False

    effective_fps = max(1.0, fps) if fps and not np.isnan(fps) else 30.0

    # Try MP4V codec (most universal on Windows/OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, effective_fps, (width, height))

    if not out.isOpened():
        # Fallback to XVID
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video_path = output_video_path.with_suffix(".avi")
        out = cv2.VideoWriter(str(output_video_path), fourcc, effective_fps, (width, height))

    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        if frame is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)

    out.release()
    return True


def extract_rgb_from_db3(
    db3_path: Path,
    output_base_dir: Path,
    rgb_topic: str = "/device_0/sensor_1/Color_0/image/data",
    overwrite: bool = False,
    generate_preview: bool = True,
    jpeg_quality: int = 95,
):
    """
    Extract RGB frames from a single ROS 2 .db3 file.

    Returns:
        dict: extraction statistics
    """
    db3_path = Path(db3_path).resolve()
    file_stem = db3_path.stem
    target_dir = output_base_dir / file_stem
    frames_dir = target_dir / "frames"
    csv_path = target_dir / "timestamps.csv"
    preview_path = target_dir / "preview.mp4"

    stats = {
        "source_file": db3_path.name,
        "target_dir": str(target_dir),
        "status": "pending",
        "frames_extracted": 0,
        "first_frame_shape": None,
        "last_frame_shape": None,
        "first_timestamp_ns": None,
        "last_timestamp_ns": None,
        "duration_seconds": 0.0,
        "calculated_fps": 0.0,
        "decoding_errors": 0,
        "preview_generated": False,
    }

    # Check if already processed
    if not overwrite and csv_path.exists() and frames_dir.exists():
        existing_frames = len(list(frames_dir.glob("frame_*.jpg")))
        if existing_frames > 0:
            print(f"Skipping '{db3_path.name}': already extracted ({existing_frames} frames). Use --overwrite to re-extract.")
            stats["status"] = "skipped"
            stats["frames_extracted"] = existing_frames
            return stats

    # Ensure directories exist
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Open SQLite in strictly read-only URI mode to guarantee zero modification
    uri = f"file:{db3_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cursor = conn.cursor()

    try:
        # 1. Identify RGB topic ID
        cursor.execute("SELECT id, type, serialization_format FROM topics WHERE name = ?", (rgb_topic,))
        topic_row = cursor.fetchone()
        if not topic_row:
            error_msg = f"RGB topic '{rgb_topic}' not found in '{db3_path.name}'"
            print(f"  [ERROR] {error_msg}")
            stats["status"] = "error: topic not found"
            return stats

        topic_id, msg_type, ser_format = topic_row

        # 2. Query messages for this topic ordered by timestamp
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
            (topic_id,),
        )
        messages = cursor.fetchall()
        total_msgs = len(messages)
        print(f"Processing '{db3_path.name}': found {total_msgs} messages on topic '{rgb_topic}'")

        if total_msgs == 0:
            stats["status"] = "empty: 0 messages"
            return stats

        csv_rows = []
        first_ts_ns = None
        last_ts_ns = None
        frame_idx = 0
        img_width = 640
        img_height = 480

        for row_idx, (ts_ns, compressed_data) in enumerate(messages, start=1):
            try:
                # 3. Decompress Zstandard payload using pyarrow streaming reader
                reader = pa.BufferReader(compressed_data)
                stream = pa.input_stream(reader, compression="zstd")
                raw_cdr = stream.read()

                # 4. Parse ROS 2 CDR Image message
                sec, nanosec, frame_id, height, width, encoding, is_bigendian, step, img_np = (
                    parse_ros2_image_cdr(raw_cdr)
                )

                img_width = width
                img_height = height

                # Record first / last shape
                if frame_idx == 0:
                    stats["first_frame_shape"] = img_np.shape
                    first_ts_ns = ts_ns
                stats["last_frame_shape"] = img_np.shape
                last_ts_ns = ts_ns

                # 5. Convert RGB to BGR for OpenCV saving
                if encoding == "rgb8":
                    bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif encoding == "bgr8":
                    bgr_frame = img_np
                else:
                    bgr_frame = img_np

                frame_idx += 1
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_filepath = frames_dir / frame_filename

                # 6. Save as JPG
                cv2.imwrite(
                    str(frame_filepath),
                    bgr_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )

                # 7. Record timestamp
                relative_sec = (ts_ns - first_ts_ns) / 1e9
                csv_rows.append([
                    frame_idx,
                    ts_ns,
                    f"{relative_sec:.3f}",
                    db3_path.name,
                    frame_filename,
                ])

            except Exception as e:
                stats["decoding_errors"] += 1
                print(f"  [ERROR] Frame {row_idx} decoding error: {e}")

        # Write timestamps.csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "frame_number",
                "timestamp_ns",
                "timestamp_seconds",
                "source_db3",
                "frame_filename",
            ])
            writer.writerows(csv_rows)

        # Compute summary statistics
        duration_s = (last_ts_ns - first_ts_ns) / 1e9 if last_ts_ns and first_ts_ns and last_ts_ns > first_ts_ns else 0.0
        fps = (frame_idx - 1) / duration_s if duration_s > 0 and frame_idx > 1 else (frame_idx / duration_s if duration_s > 0 else 0.0)

        stats["status"] = "success"
        stats["frames_extracted"] = frame_idx
        stats["first_timestamp_ns"] = first_ts_ns
        stats["last_timestamp_ns"] = last_ts_ns
        stats["duration_seconds"] = duration_s
        stats["calculated_fps"] = fps

        # 8. Create preview video if requested
        if generate_preview and frame_idx > 0:
            preview_ok = create_preview_video(
                frames_dir, preview_path, fps=fps, width=img_width, height=img_height
            )
            stats["preview_generated"] = preview_ok
            if preview_ok:
                print(f"  Preview video generated: {preview_path}")

        print(f"  Extracted {frame_idx} frames | Duration: {duration_s:.2f}s | FPS: {fps:.2f} | Errors: {stats['decoding_errors']}")

    finally:
        conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract RGB frames from ROS 2 .db3 bag files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Path to folder containing .db3 files (default: 'input')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted",
        help="Path to output base directory (default: 'extracted')",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific .db3 file name or path to process (optional). If not specified, all files are processed.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/device_0/sensor_1/Color_0/image/data",
        help="RGB image topic name (default: '/device_0/sensor_1/Color_0/image/data')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing extracted frames and timestamps.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable automatic preview.mp4 generation.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            # Check in input_dir or cwd
            if (input_dir / file_path.name).exists():
                file_path = input_dir / file_path.name
            else:
                file_path = Path.cwd() / file_path

        if not file_path.exists():
            print(f"[ERROR] Specified file '{file_path}' does not exist.")
            sys.exit(1)
        files = [file_path]
    else:
        files = sorted(list(input_dir.glob("*.db3")))

    print(f"Found {len(files)} file(s) to process.")
    print(f"Output directory: {output_dir}")
    print(f"RGB Topic: {args.topic}")
    print("=" * 70)

    results = []
    for f in files:
        res = extract_rgb_from_db3(
            db3_path=f,
            output_base_dir=output_dir,
            rgb_topic=args.topic,
            overwrite=args.overwrite,
            generate_preview=not args.no_preview,
            jpeg_quality=args.quality,
        )
        results.append(res)
        print("-" * 70)

    print(f"Summary: Processed {len(results)} file(s).")


if __name__ == "__main__":
    main()

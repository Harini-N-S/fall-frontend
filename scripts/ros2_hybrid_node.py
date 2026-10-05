#!/usr/bin/env python3
"""
ros2_hybrid_node.py - Real-Time ROS 2 Hybrid Fall Detection Node & Offline Replay Engine.

Supports two operational modes:
    1. ROS 2 Live Node Mode:
       Active when rclpy and ROS 2 middleware are installed.
       Subscribes to sensor_msgs/msg/Image and publishes confirmed alerts to /fall_detection/alerts.
    2. Offline Bag-Replay / Dry-Run Mode:
       Reads directly from ROS 2 .db3 bag files or extracted videos/images without requiring ROS 2.
       Streams frames frame-by-frame with exact real-time semantics into RealtimeHybridDetector.

Usage:
    # Live ROS 2 Node:
    python scripts/ros2_hybrid_node.py --live

    # Replay ROS 2 .db3 Bag directly:
    python scripts/ros2_hybrid_node.py --bag input/20260828_134619.db3

    # Fast validation across multiple bags:
    python scripts/ros2_hybrid_node.py --replay-all
"""

import sys
import os
import argparse
import time
import json
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import cv2
import yaml

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from alert_manager import createFallAlert
from patient_config import get_patient_info

from hybrid_detector import RealtimeHybridDetector
from extract_frames import parse_ros2_image_cdr

# Optional PyArrow for Zstandard decompression of .db3 payloads
try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# Optional ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import String as RosString
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class Ros2HybridFallDetectorNode:
    """ROS 2 Node wrapper for RealtimeHybridDetector."""

    def __init__(self, detector: RealtimeHybridDetector, image_topic: str, alert_topic: str):
        if not HAS_ROS2:
            raise RuntimeError("rclpy / ROS 2 is not available in the current environment.")

        # Subclassing Node dynamically
        class _NodeImpl(Node):
            def __init__(self, outer):
                super().__init__("hybrid_fall_detector_node")
                self.outer = outer
                self.bridge = CvBridge()
                self.subscription = self.create_subscription(
                    RosImage,
                    image_topic,
                    self.image_callback,
                    qos_profile_sensor_data,
                )
                self.publisher = self.create_publisher(RosString, alert_topic, 10)
                self.get_logger().info(f"Hybrid Fall Detector subscribed to {image_topic}")
                self.get_logger().info(f"Publishing confirmed alerts to {alert_topic}")

            def image_callback(self, msg: RosImage):
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    res = self.outer.detector.process_frame(cv_image, ts)

                    if res["confirmed_fall"]:
                        alert_json = json.dumps(res["confirmed_fall"])
                        alert_msg = RosString()
                        alert_msg.data = alert_json
                        self.publisher.publish(alert_msg)
                        self.get_logger().warn(f"CONFIRMED FALL ALERT: {alert_json}")

                    if res["rejected_alarm"]:
                        self.get_logger().info(f"Rejected ML false alarm: {res['rejected_alarm']['rejection_reason']}")

                except Exception as e:
                    self.get_logger().error(f"Error processing frame: {e}")

        self.detector = detector
        self.node = _NodeImpl(self)

    def spin(self):
        rclpy.spin(self.node)


def replay_db3_bag(
    db3_path: Path,
    detector: RealtimeHybridDetector,
    rgb_topic: str = "/device_0/sensor_1/Color_0/image/data",
    simulate_realtime: bool = False,
    max_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Replays a ROS 2 .db3 file directly through the RealtimeHybridDetector.
    """
    db3_path = Path(db3_path).resolve()
    session_id = db3_path.stem
    detector.reset(session_id=session_id)

    uri = f"file:{db3_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cursor = conn.cursor()

    # Identify topic
    cursor.execute("SELECT id FROM topics WHERE name = ?", (rgb_topic,))
    topic_row = cursor.fetchone()
    if not topic_row:
        raise ValueError(f"Topic '{rgb_topic}' not found in {db3_path.name}")
    topic_id = topic_row[0]

    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (topic_id,))
    messages = cursor.fetchall()
    conn.close()

    total_msgs = len(messages)
    print(f"\n[Replay] Processing '{db3_path.name}' ({total_msgs} messages)...")

    results = []
    first_ts = None
    prev_wall_time = time.time()

    for idx, (ts_ns, compressed_data) in enumerate(messages, start=1):
        if max_frames and idx > max_frames:
            break

        # Decompress Zstd
        reader = pa.BufferReader(compressed_data)
        stream = pa.input_stream(reader, compression="zstd")
        raw_cdr = stream.read()

        # Parse ROS 2 CDR Image message
        sec, nanosec, frame_id, height, width, encoding, is_bigendian, step, img_np = parse_ros2_image_cdr(raw_cdr)
        ts_sec = sec + nanosec * 1e-9

        if first_ts is None:
            first_ts = ts_sec

        # Optional real-time rate limiter
        if simulate_realtime and idx > 1:
            dt = ts_sec - prev_ts
            time_spent = time.time() - prev_wall_time
            if dt > time_spent:
                time.sleep(dt - time_spent)

        prev_wall_time = time.time()
        prev_ts = ts_sec

        # Process frame
        res = detector.process_frame(img_np, ts_sec, frame_number=idx)
        results.append(res)

        if res["confirmed_fall"]:
            ev = res["confirmed_fall"]
            print(f"  >>> [CONFIRMED FALL] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s | Confirmed At: {ev['heuristic_confirmation_time']:.2f}s | Lead Time: +{ev['advance_lead_time_seconds']:.2f}s | Max Prob: {ev['peak_ml_probability']:.3f}")
            p_info = get_patient_info(ev['session_id'])
            createFallAlert(
                patientId=p_info["patient_id"],
                roomId=p_info["room"],
                timestamp=ts_sec,
                fall_details={
                    "session_id": ev['session_id'],
                    "lead_time": ev.get("advance_lead_time_seconds"),
                    "peak_ml_probability": ev.get("peak_ml_probability"),
                },
            )

        if res["rejected_alarm"]:
            ev = res["rejected_alarm"]
            print(f"  --- [REJECTED ALARM] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s - {ev['ml_end_time']:.2f}s | {ev['rejection_reason']}")

    return results


def replay_features_stream(
    rec_stem: str,
    detector: RealtimeHybridDetector,
) -> List[Dict[str, Any]]:
    """
    Replays a recording from pre-extracted features in data/features/<rec_stem>.npz.
    Streams feature vectors and kinematics frame-by-frame with exact real-time semantics.
    """
    npz_path = BASE_DIR / "data" / "features" / f"{rec_stem}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    data = np.load(npz_path)
    feat_matrix = data["features"]      # (N, 62)
    kin_matrix = data["kinematics"]    # (N, 7)
    timestamps = data["timestamps"]    # (N,)
    frame_numbers = data["frame_numbers"]  # (N,)

    detector.reset(session_id=rec_stem)
    total_frames = len(timestamps)
    print(f"\n[Replay] Streaming '{rec_stem}' ({total_frames} frames)...", flush=True)

    results = []
    for idx in range(total_frames):
        ts = float(timestamps[idx])
        f_num = int(frame_numbers[idx])
        feat_vec = feat_matrix[idx]
        kin_vec = kin_matrix[idx]

        kin_dict = detector.kinematics_from_array(kin_vec, ts)
        res = detector.process_feature_step(feat_vec, kin_dict, ts, frame_number=f_num)
        results.append(res)

        if res["confirmed_fall"]:
            ev = res["confirmed_fall"]
            print(f"  >>> [CONFIRMED FALL] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s | Confirmed At: {ev['heuristic_confirmation_time']:.2f}s | Lead Time: +{ev['advance_lead_time_seconds']:.2f}s | Max Prob: {ev['peak_ml_probability']:.3f}", flush=True)

        if res["rejected_alarm"]:
            ev = res["rejected_alarm"]
            print(f"  --- [REJECTED ALARM] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s - {ev['ml_end_time']:.2f}s | {ev['rejection_reason']}", flush=True)

    # If provisional event remains at end of stream, resolve as rejected
    if detector.active_provisional_event:
        p = detector.active_provisional_event
        rej = {
            "event_id": f"REJ_{detector.current_session_id}_{len(detector.event_log)+1:03d}",
            "session_id": detector.current_session_id,
            "status": "REJECTED_ML_FALSE_ALARM",
            "ml_trigger_time": round(p["ml_trigger_time"], 3),
            "ml_end_time": round(p["ml_end_time"], 3),
            "expiration_time": round(p["ml_end_time"] + detector.confirmation_window_s, 3),
            "peak_ml_probability": round(max(p["probabilities"]), 4),
            "rejection_reason": f"Recording ended without heuristic confirmation within W={detector.confirmation_window_s:.1f}s",
        }
        detector.event_log.append(rej)
        print(f"  --- [REJECTED ALARM] Session: {rej['session_id']} | ML Trigger: {rej['ml_trigger_time']:.2f}s - {rej['ml_end_time']:.2f}s | {rej['rejection_reason']}", flush=True)
        detector.active_provisional_event = None

    return results


def replay_extracted_recording(
    rec_stem: str,
    detector: RealtimeHybridDetector,
    simulate_realtime: bool = False,
) -> List[Dict[str, Any]]:
    """
    Replays from extracted preview video and timestamps.csv (faster if already extracted).
    """
    rec_dir = BASE_DIR / "extracted" / rec_stem
    csv_path = rec_dir / "timestamps.csv"
    vid_path = rec_dir / "preview.mp4"

    if not csv_path.exists() or not vid_path.exists():
        # Fallback to .db3
        db3_path = BASE_DIR / "input" / f"{rec_stem}.db3"
        return replay_db3_bag(db3_path, detector, simulate_realtime=simulate_realtime)

    import csv
    timestamps = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            timestamps.append(float(r["timestamp_seconds"]))

    cap = cv2.VideoCapture(str(vid_path))
    detector.reset(session_id=rec_stem)

    results = []
    f_idx = 0
    total_frames = len(timestamps)
    print(f"\n[Replay] Processing '{rec_stem}' ({total_frames} frames)...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret or f_idx >= total_frames:
            break

        ts = timestamps[f_idx]
        f_idx += 1

        res = detector.process_frame(frame, ts, frame_number=f_idx)
        results.append(res)

        if f_idx % 100 == 0:
            print(f"  [Progress] Frame {f_idx}/{total_frames} ({ts:.1f}s)...", flush=True)

        if res["confirmed_fall"]:
            ev = res["confirmed_fall"]
            print(f"  >>> [CONFIRMED FALL] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s | Confirmed At: {ev['heuristic_confirmation_time']:.2f}s | Lead Time: +{ev['advance_lead_time_seconds']:.2f}s | Max Prob: {ev['peak_ml_probability']:.3f}", flush=True)

        if res["rejected_alarm"]:
            ev = res["rejected_alarm"]
            print(f"  --- [REJECTED ALARM] Session: {ev['session_id']} | ML Trigger: {ev['ml_trigger_time']:.2f}s - {ev['ml_end_time']:.2f}s | {ev['rejection_reason']}", flush=True)

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="ROS 2 Hybrid Fall Detector & Replay Engine")
    parser.add_argument("--config", type=str, default=str(BASE_DIR / "config" / "hybrid_detector_config.yaml"), help="Config file")
    parser.add_argument("--bag", type=str, default=None, help="Path to ROS 2 .db3 bag to replay")
    parser.add_argument("--stem", type=str, default=None, help="Recording stem to replay (e.g. 20260828_134619)")
    parser.add_argument("--replay-all", action="store_true", help="Run verification replay across all 6 test recordings")
    parser.add_argument("--use-video", action="store_true", help="Run YOLOv8 from video instead of feature stream")
    parser.add_argument("--live", action="store_true", help="Launch live ROS 2 node")
    parser.add_argument("--output-json", type=str, default="results/step12_realtime_event_log.json", help="Path to save event logs")
    args = parser.parse_args()

    detector = RealtimeHybridDetector(config_path=Path(args.config))

    if args.live:
        if not HAS_ROS2:
            print("[ERROR] rclpy is not installed. Live ROS 2 mode cannot start.")
            sys.exit(1)
        rclpy.init()
        node = Ros2HybridFallDetectorNode(
            detector,
            image_topic=detector.cfg["ros2"]["image_topic"],
            alert_topic=detector.cfg["ros2"]["alert_topic"],
        )
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()

    elif args.replay_all:
        test_recordings = [
            "20260817_153818",
            "20260817_160451",
            "20260828_134619",
            "20260828_133409",
            "20260817_153540",
            "20260828_135543",
        ]
        all_logs = []
        for stem in test_recordings:
            if args.use_video:
                replay_extracted_recording(stem, detector)
            else:
                replay_features_stream(stem, detector)
            all_logs.extend(detector.event_log)

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2)
        print(f"\n[Complete] Replay finished across 6 test recordings. Event logs saved to: {out_path}", flush=True)

    elif args.bag:
        bag_path = Path(args.bag)
        replay_db3_bag(bag_path, detector)
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detector.event_log, f, indent=2)
        print(f"\nEvent log saved to: {out_path}", flush=True)

    elif args.stem:
        if args.use_video:
            replay_extracted_recording(args.stem, detector)
        else:
            replay_features_stream(args.stem, detector)
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detector.event_log, f, indent=2)
        print(f"\nEvent log saved to: {out_path}", flush=True)

    else:
        print("Specify --live, --bag <path>, --stem <stem>, or --replay-all. Run with --help for options.", flush=True)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
run_hybrid_pipeline.py - Unified Production Entrypoint for Hybrid Fall Detection.
================================================================================
Supports multiple production deployment modes:
  1. ROS 2 Live Node:
     python scripts/run_hybrid_pipeline.py --live-ros2
  2. Rosbag Replay (.db3):
     python scripts/run_hybrid_pipeline.py --replay-bag input/20260817_153818.db3 --output-json results/alert.json
  3. Feature Stream Replay:
     python scripts/run_hybrid_pipeline.py --replay-features 20260817_153818
  4. Video / Webcam Stream:
     python scripts/run_hybrid_pipeline.py --video path/to/video.mp4 --visualize
     python scripts/run_hybrid_pipeline.py --camera 0 --visualize
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import cv2
import yaml

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from hybrid_detector import RealtimeHybridDetector
from ros2_hybrid_node import replay_db3_bag, HAS_ROS2
from alert_manager import createFallAlert
from patient_config import get_patient_info

if HAS_ROS2:
    from ros2_hybrid_node import Ros2HybridFallDetectorNode, rclpy


def replay_feature_stream(
    feat_path: Path,
    detector: RealtimeHybridDetector,
    simulate_realtime: bool = False,
) -> List[Dict[str, Any]]:
    """
    Streams a precomputed feature file (.npz) frame-by-frame with full real-time semantics.
    """
    stem = feat_path.stem
    npz = np.load(feat_path)
    feats = npz["features"]
    kins = npz["kinematics"]
    ts = npz["timestamps"]
    f_nums = npz["frame_numbers"]

    total_frames = len(ts)
    detector.reset(session_id=stem)
    print(f"\n[Feature Stream] Processing '{stem}' ({total_frames} frames)...")

    results = []
    prev_wall = time.time()
    for i in range(total_frames):
        t_frame = float(ts[i])
        fn = int(f_nums[i])

        if simulate_realtime and i > 0:
            dt = t_frame - float(ts[i-1])
            elapsed = time.time() - prev_wall
            if dt > elapsed:
                time.sleep(dt - elapsed)
        prev_wall = time.time()

        kd = detector.kinematics_from_array(kins[i], t_frame)
        res = detector.process_feature_step(feats[i], kd, t_frame, frame_number=fn)
        results.append(res)

        if res.get("confirmed_fall"):
            c = res["confirmed_fall"]
            print(f"  >>> [CONFIRMED FALL] Session: {stem} | Alert: {c['final_alert_time']:.2f}s | Confirmed: {c['heuristic_confirmation_time']:.2f}s | Lead: +{c['advance_lead_time_seconds']:.2f}s")
            p_info = get_patient_info(stem)
            createFallAlert(
                patientId=p_info["patient_id"],
                roomId=p_info["room"],
                timestamp=t_frame,
                fall_details={
                    "session_id": stem,
                    "lead_time": c.get("advance_lead_time_seconds"),
                    "peak_ml_probability": c.get("peak_ml_probability"),
                    "heuristic_confirmation_time": c.get("heuristic_confirmation_time", t_frame),
                },
            )

        if res.get("rejected_alarm"):
            r = res["rejected_alarm"]
            print(f"  --- [REJECTED ALARM] Session: {stem} | Trigger: {r['ml_trigger_time']:.2f}s | {r['rejection_reason']}")

    # End of recording cleanup
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
        print(f"  --- [REJECTED ALARM] Session: {stem} | Trigger: {rej['ml_trigger_time']:.2f}s | {rej['rejection_reason']}")
        detector.active_provisional_event = None

    return results


def draw_detection_overlay(
    img: np.ndarray,
    res: Dict[str, Any],
    fps_estimate: float = 0.0,
    is_live: bool = False,
) -> np.ndarray:
    """
    Renders bounding box, skeleton, FSM state, ML probability, and alerts.
    """
    out = img.copy()
    h, w = out.shape[:2]

    fsm_state = res.get("fsm_state", "NORMAL")
    target_id = res.get("track_id")
    target_str = f"#{target_id}" if target_id is not None else "Searching..."

    COCO_LIMBS = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]

    def _draw_skeleton(kpts, limb_color=(255, 140, 0), joint_color=(0, 255, 255)):
        if kpts is not None and len(kpts) >= 17:
            for p1, p2 in COCO_LIMBS:
                if p1 < len(kpts) and p2 < len(kpts):
                    pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                    pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(out, pt1, pt2, limb_color, 2)
            for pt in kpts:
                x, y = int(pt[0]), int(pt[1])
                if x > 0 and y > 0:
                    cv2.circle(out, (x, y), 4, joint_color, -1)

    # Draw all detected persons with clear target vs bystander demarcation
    all_boxes = res.get("all_boxes")
    if all_boxes:
        # Draw pose keypoints and skeleton for all detected persons
        for b_info in all_boxes:
            person_kpts = b_info.get("keypoints")
            if person_kpts is None and b_info.get("is_target", False):
                person_kpts = res.get("keypoints")
            _draw_skeleton(person_kpts)

        # Draw bounding boxes and person annotations
        for b_info in all_boxes:
            b = b_info["box"]
            tid = b_info["track_id"]
            is_target = b_info.get("is_target", False)
            bx1, by1, bx2, by2 = [int(v) for v in b]
            if is_target:
                # Target person: highlighted green box
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                label = f"Target #{tid} [{fsm_state}]" if tid is not None else f"Target [{fsm_state}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(out, (bx1, max(0, by1 - th - 8)), (bx1 + tw + 6, max(th + 8, by1)), (20, 20, 20), -1)
                cv2.putText(out, label, (bx1 + 3, max(th + 2, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Clearly visible contrasting box for unmonitored person (Cyan/Amber in BGR: (255, 200, 0))
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                label = f"Track #{tid} [Unmonitored]" if tid is not None else "Track [Unmonitored]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(out, (bx1, max(0, by1 - th - 8)), (bx1 + tw + 6, max(th + 8, by1)), (40, 40, 40), -1)
                cv2.putText(out, label, (bx1 + 3, max(th + 2, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 1, cv2.LINE_AA)
    else:
        # Fallback to single box and skeleton if present
        box = res.get("box")
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _draw_skeleton(res.get("keypoints"))

    # Header HUD: TARGET ID & FSM STATE
    state_color = (
        (0, 255, 0) if fsm_state == "NORMAL"
        else (255, 200, 0) if fsm_state == "SITTING"
        else (0, 165, 255) if fsm_state == "FALLING"
        else (0, 0, 255)
    )
    
    cv2.rectangle(out, (0, 0), (w, 65), (20, 20, 20), -1)
    hud_line1 = f"TARGET ID: {target_str}   |   FSM: {fsm_state}"
    cv2.putText(out, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    ml_prob = res.get("ml_probability")
    prob_str = f"ML Prob: {ml_prob:.3f}" if ml_prob is not None else "ML: Buffering..."
    cv2.putText(out, prob_str, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    
    ts = res.get("timestamp_seconds", 0.0)
    if is_live:
        cv2.circle(out, (w - 275, 20), 5, (0, 0, 255), -1)
        cv2.putText(out, "LIVE D435", (w - 263, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"T: {ts:.1f}s | FPS: {fps_estimate:.1f}", (w - 175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    else:
        cv2.putText(out, f"Time: {ts:.2f}s | FPS: {fps_estimate:.1f}", (w - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Confirmed Alert Banner
    if res.get("confirmed_fall"):
        alert = res["confirmed_fall"]
        cv2.rectangle(out, (0, h - 70), (w, h), (0, 0, 220), -1)
        lead = alert.get("advance_lead_time_seconds", 0.0)
        cv2.putText(out, f"*** CONFIRMED FALL ALERT (Lead Time: +{lead:.2f}s) ***", (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    return out


def run_video_or_camera(
    source: Any,
    detector: RealtimeHybridDetector,
    visualize: bool = False,
    output_json: Optional[Path] = None,
    max_frames: Optional[int] = None,
    save_viz: Optional[Path] = None,
):
    """
    Runs RealtimeHybridDetector on a standard OpenCV video file or webcam.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    session_id = Path(str(source)).stem if isinstance(source, (str, Path)) else f"camera_{source}"
    detector.reset(session_id=session_id)
    print(f"\n[Video/Stream] Processing '{session_id}' (Source FPS: {fps:.1f})...")

    frame_idx = 0
    t_start = time.time()
    results = []
    saved_snapshot = False

    video_writer = None
    if save_viz and save_viz.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        save_viz.parent.mkdir(parents=True, exist_ok=True)
        w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(save_viz), fourcc, fps, (w_img, h_img))
        print(f"  [Recording] Saving annotated tracking video to: {save_viz}")

    alerted_event_ids = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            ts = (frame_idx - 1) / fps
            t0 = time.perf_counter()
            res = detector.process_frame(frame, timestamp_s=ts, frame_number=frame_idx)
            proc_dt = time.perf_counter() - t0
            inst_fps = 1.0 / max(proc_dt, 1e-6)
            results.append(res)

            if res.get("confirmed_fall"):
                c = res["confirmed_fall"]
                ev_id = c.get("event_id") or f"{session_id}_{c.get('heuristic_confirmation_time', ts):.2f}"
                if ev_id not in alerted_event_ids:
                    alerted_event_ids.add(ev_id)
                    print(f"  [ALERT] CONFIRMED FALL at {ts:.2f}s! (Lead: +{c.get('advance_lead_time_seconds', 0.0):.2f}s)", flush=True)
                    p_info = get_patient_info(session_id)
                    createFallAlert(
                        patientId=p_info["patient_id"],
                        roomId=p_info["room"],
                        timestamp=ts,
                        fall_details={
                            "session_id": session_id,
                            "lead_time": c.get("advance_lead_time_seconds"),
                            "peak_ml_probability": c.get("peak_ml_probability"),
                            "heuristic_confirmation_time": c.get("heuristic_confirmation_time", ts),
                        },
                    )

            if res.get("rejected_alarm"):
                r = res["rejected_alarm"]
                print(f"  [REJECT] ML alarm rejected at {ts:.2f}s: {r.get('rejection_reason')}", flush=True)

            if visualize or save_viz:
                disp = draw_detection_overlay(frame, res, fps_estimate=inst_fps)
                
                if video_writer is not None:
                    video_writer.write(disp)
                elif save_viz and (not saved_snapshot or res.get("confirmed_fall")):
                    save_viz.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_viz), disp)
                    saved_snapshot = True
                
                if visualize:
                    cv2.imshow("Hybrid Fall Detector", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[User Interrupt] Stopping visualization.")
                        break

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"  [Recorded] Annotated video saved successfully ({frame_idx} frames).")
        if visualize:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    confirmed_falls = [e for e in detector.event_log if e.get("status") == "CONFIRMED_FALL"]
    rejected_alarms = [e for e in detector.event_log if e.get("status") == "REJECTED_ML_FALSE_ALARM"]
    print(f"  Confirmed Falls: {len(confirmed_falls)}")
    print(f"  Rejected Alarms: {len(rejected_alarms)}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(detector.event_log, f, indent=2)
        print(f"  Saved event log to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Production Hybrid Fall Detection Runner (Strategy H5)"
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config" / "hybrid_detector_config.yaml", help="Path to config YAML")
    parser.add_argument("--live-ros2", action="store_true", help="Launch live ROS 2 node")
    parser.add_argument("--replay-bag", type=Path, help="Replay a ROS 2 SQLite3 (.db3) bag file directly")
    parser.add_argument("--replay-features", type=str, help="Replay pre-extracted feature stream by recording stem, or 'all'")
    parser.add_argument("--video", type=Path, help="Run detection on a video file (.mp4, .avi, etc.)")
    parser.add_argument("--camera", type=int, help="Run live detection on webcam device ID (e.g. 0)")
    parser.add_argument("--realsense", action="store_true", help="Launch live Intel RealSense D435 camera mode")
    parser.add_argument("--realsense-fps", type=int, default=30, help="Camera target FPS for RealSense stream (default: 30)")
    parser.add_argument("--realsense-serial", type=str, default=None, help="Optional RealSense device serial number")
    parser.add_argument("--realsense-mock", type=Path, default=None, help="Simulate RealSense stream using video file for testing")
    parser.add_argument("--visualize", action="store_true", help="Display live video window with skeleton and fall alerts")
    parser.add_argument("--save-viz", type=Path, help="Save sample visualization snapshot to image path")
    parser.add_argument("--output-json", type=Path, help="Output file to save detected events JSON")
    parser.add_argument("--max-frames", type=int, help="Optional limit on number of frames to process")
    parser.add_argument("--simulate-realtime", action="store_true", help="Throttle playback to original frame timestamps")

    args = parser.parse_args()

    detector = RealtimeHybridDetector(config_path=args.config)

    # 1. Live ROS 2 Node
    if args.live_ros2:
        if not HAS_ROS2:
            print("[ERROR] ROS 2 (rclpy) is not installed in this environment.")
            sys.exit(1)
        print("\n[ROS 2] Initializing Hybrid Fall Detector ROS 2 Node...")
        rclpy.init()
        node = Ros2HybridFallDetectorNode(detector=detector)
        try:
            node.spin()
        except KeyboardInterrupt:
            print("\n[ROS 2] Shutting down node.")
        finally:
            rclpy.shutdown()
        return

    # 2. Replay Rosbag
    elif args.replay_bag:
        if not args.replay_bag.exists():
            print(f"[ERROR] Bag file does not exist: {args.replay_bag}")
            sys.exit(1)
        res = replay_db3_bag(
            args.replay_bag,
            detector,
            simulate_realtime=args.simulate_realtime,
            max_frames=args.max_frames,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(detector.event_log, f, indent=2)
            print(f"\n[Saved] Event log saved to: {args.output_json}")
        return

    # 3. Feature Stream Replay
    elif args.replay_features:
        if args.replay_features == "all":
            feat_files = sorted((BASE_DIR / "data" / "features").glob("*.npz"))
            print(f"\n[Feature Replay] Running across all {len(feat_files)} feature files...")
            for ff in feat_files:
                replay_feature_stream(ff, detector, simulate_realtime=args.simulate_realtime)
        else:
            feat_file = BASE_DIR / "data" / "features" / f"{args.replay_features}.npz"
            if not feat_file.exists():
                print(f"[ERROR] Feature file not found: {feat_file}")
                sys.exit(1)
            replay_feature_stream(feat_file, detector, simulate_realtime=args.simulate_realtime)
            
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(detector.event_log, f, indent=2)
            print(f"\n[Saved] Event log saved to: {args.output_json}")
        return

    # 4. Video File
    elif args.video:
        run_video_or_camera(
            str(args.video),
            detector,
            visualize=args.visualize,
            output_json=args.output_json,
            max_frames=args.max_frames,
            save_viz=args.save_viz,
        )
        return

    # 5. Webcam Device
    elif args.camera is not None:
        run_video_or_camera(
            args.camera,
            detector,
            visualize=args.visualize,
            output_json=args.output_json,
            max_frames=args.max_frames,
            save_viz=args.save_viz,
        )
        return

    # 6. Live Intel RealSense D435 Mode
    elif args.realsense:
        from run_realsense_live import run_realsense_stream
        run_realsense_stream(
            detector,
            width=640,
            height=480,
            fps=args.realsense_fps,
            visualize=args.visualize if ("--visualize" in sys.argv or "-v" in sys.argv) else True,
            output_json=args.output_json,
            max_frames=args.max_frames,
            save_viz=args.save_viz,
            serial=args.realsense_serial,
            mock_source=args.realsense_mock,
        )
        return

    else:
        parser.print_help()
        print("\nExample commands:")
        print("  python scripts/run_hybrid_pipeline.py --realsense")
        print("  python scripts/run_hybrid_pipeline.py --replay-bag input/20260817_153818.db3 --output-json results/test.json")
        print("  python scripts/run_hybrid_pipeline.py --replay-features 20260817_153818")
        print("  python scripts/run_hybrid_pipeline.py --replay-features all")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_realsense_live.py - Dedicated Live Intel RealSense D435 Fall Detection Runner.
=================================================================================
Captures live RGB frames from an Intel RealSense D435 camera (640x480 @ 30 FPS)
and feeds them directly into the existing production RealtimeHybridDetector pipeline.

Key Guarantees:
  1. Frozen Model & Logic: Uses existing YOLOv8-Pose, ByteTrack target lock,
     biomechanical heuristics, FSM, and frozen FallGRUClassifier without modification.
  2. Multi-Person Visualization: Full 17-keypoint skeletons for all detected persons;
     green highlighted box for primary target; cyan/amber unmonitored styling for bystanders.
  3. Escalation Integration: Automatically forwards confirmed fall events to the
     FastAPI Nurse Dashboard via existing createFallAlert().
  4. Robust Lifecycle: Safe camera initialization, graceful error reporting if
     device is unplugged, and clean resource release on 'q' keypress.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

# Project root setup
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hybrid_detector import RealtimeHybridDetector
from run_hybrid_pipeline import draw_detection_overlay
from alert_manager import createFallAlert
from patient_config import get_patient_info


def check_realsense_availability() -> Tuple[bool, str, List[Any]]:
    """
    Checks if pyrealsense2 is installed and whether a RealSense device is physically connected.
    
    Returns:
        (is_available, message, devices_list)
    """
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        msg = (
            f"pyrealsense2 is not installed or unavailable: {e}\n"
            f"Please install it using: pip install pyrealsense2"
        )
        return False, msg, []

    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        dev_count = len(devices)
        if dev_count == 0:
            return False, "Intel RealSense camera not detected.", []
        return True, f"Found {dev_count} RealSense device(s).", list(devices)
    except Exception as ex:
        return False, f"Error querying RealSense devices: {ex}", []


def init_realsense_pipeline(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any], int, int, int]:
    """
    Initializes and starts the pyrealsense2 pipeline for RGB streaming on the D435.

    Returns:
        (pipeline, profile, actual_width, actual_height, actual_fps)
        or (None, None, 0, 0, 0) upon failure.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        print(f"\n[ERROR] pyrealsense2 is not installed: {e}")
        print("Please install pyrealsense2: pip install pyrealsense2\n")
        return None, None, 0, 0, 0

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("\n[ERROR] Intel RealSense camera not detected.")
        print("Please verify that your Intel RealSense D435 is plugged into a USB 3.0 port.\n")
        return None, None, 0, 0, 0

    selected_dev = None
    if serial:
        for d in devices:
            if d.get_info(rs.camera_info.serial_number) == serial:
                selected_dev = d
                break
        if selected_dev is None:
            print(f"\n[ERROR] Intel RealSense camera with serial '{serial}' not found.")
            return None, None, 0, 0, 0
    else:
        selected_dev = devices[0]

    dev_name = selected_dev.get_info(rs.camera_info.name)
    dev_sn = selected_dev.get_info(rs.camera_info.serial_number)
    usb_desc = (
        selected_dev.get_info(rs.camera_info.usb_type_descriptor)
        if hasattr(rs.camera_info, "usb_type_descriptor") and selected_dev.supports(rs.camera_info.usb_type_descriptor)
        else "N/A"
    )
    print(f"\n[RealSense] Initializing {dev_name} (S/N: {dev_sn}, USB: {usb_desc})...")

    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    # Frame rates to try: preferred FPS first, then fallback to 15 or 30
    rates_to_try = [fps]
    for fallback_rate in (30, 15, 60):
        if fallback_rate not in rates_to_try:
            rates_to_try.append(fallback_rate)

    profile = None
    actual_fps = fps

    for rate in rates_to_try:
        try:
            config.disable_all_streams()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, rate)
            profile = pipeline.start(config)
            actual_fps = rate
            break
        except Exception as ex:
            if rate == rates_to_try[-1]:
                print(f"\n[ERROR] Failed to start Intel RealSense stream: {ex}")
                print("Please check that no other software is using the camera and reconnect the device.\n")
                return None, None, 0, 0, 0
            continue

    col_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    act_w = col_stream.width()
    act_h = col_stream.height()
    act_fps = col_stream.fps()
    print(f"[RealSense] RGB stream active: {act_w}x{act_h} @ {act_fps} FPS (format: BGR8)")

    return pipeline, profile, act_w, act_h, act_fps


def run_realsense_stream(
    detector: RealtimeHybridDetector,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    visualize: bool = True,
    output_json: Optional[Path] = None,
    max_frames: Optional[int] = None,
    save_viz: Optional[Path] = None,
    session_id: str = "live_realsense",
    serial: Optional[str] = None,
    mock_source: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Main live frame acquisition and detection loop for Intel RealSense D435.
    
    Processes every incoming RGB frame through RealtimeHybridDetector:
      - YOLOv8-Pose tracking
      - Target lock and 62-d feature extraction
      - Biomechanical FSM + Frozen GRU
      - createFallAlert() on confirmed fall
      - OpenCV display with live HUD, skeletons, and track badges
    """
    pipeline = None
    cap_mock = None

    if mock_source is not None:
        print(f"\n[Live Stream] Running in SIMULATED camera mode from source: {mock_source}")
        cap_mock = cv2.VideoCapture(str(mock_source) if isinstance(mock_source, Path) else mock_source)
        if not cap_mock.isOpened():
            raise RuntimeError(f"Failed to open mock video stream source: {mock_source}")
        src_fps = cap_mock.get(cv2.CAP_PROP_FPS)
        act_fps = int(round(src_fps)) if (src_fps and not np.isnan(src_fps) and src_fps > 0) else fps
        act_w = int(cap_mock.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        act_h = int(cap_mock.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    else:
        pipeline, profile, act_w, act_h, act_fps = init_realsense_pipeline(
            width=width,
            height=height,
            fps=fps,
            serial=serial,
        )
        if pipeline is None:
            # Camera initialization failed or not detected; already reported clear error.
            return []

    detector.reset(session_id=session_id)
    print(f"[Live Pipeline] Detector initialized for session '{session_id}' (Target: {act_w}x{act_h} @ {act_fps} FPS)")
    if visualize:
        print("[Live Pipeline] Display window active. Press 'q' in the window to stop safely.\n")

    video_writer = None
    if save_viz and save_viz.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        save_viz.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(save_viz), fourcc, float(act_fps), (act_w, act_h))
        print(f"[Recording] Saving annotated live stream to: {save_viz}")

    frame_idx = 0
    t_start = time.perf_counter()
    first_frame_hw_ts: Optional[float] = None
    results = []
    alerted_event_ids = set()
    saved_snapshot = False

    fps_history = []

    try:
        while True:
            if pipeline is not None:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except Exception as ex:
                    print(f"\n[RealSense Warning] Frame timeout or read error: {ex}")
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())

                # RealSense frame hardware/system timestamp (in ms)
                hw_ts = color_frame.get_timestamp()
                if first_frame_hw_ts is None:
                    first_frame_hw_ts = hw_ts
                ts = (hw_ts - first_frame_hw_ts) / 1000.0

            elif cap_mock is not None:
                ret, frame = cap_mock.read()
                if not ret:
                    break
                ts = (frame_idx) / float(act_fps)
            else:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            t0 = time.perf_counter()
            res = detector.process_frame(frame, timestamp_s=ts, frame_number=frame_idx)
            proc_dt = time.perf_counter() - t0

            inst_fps = 1.0 / max(proc_dt, 1e-6)
            fps_history.append(inst_fps)
            if len(fps_history) > 60:
                fps_history.pop(0)
            avg_recent_fps = float(np.mean(fps_history))

            results.append(res)

            # Confirmed Fall Alert -> Dashboard Escalation
            if res.get("confirmed_fall"):
                c = res["confirmed_fall"]
                ev_id = c.get("event_id") or f"{session_id}_{c.get('heuristic_confirmation_time', ts):.2f}"
                if ev_id not in alerted_event_ids:
                    alerted_event_ids.add(ev_id)
                    lead_time = c.get("advance_lead_time_seconds", 0.0)
                    print(
                        f"\n  >>> [CONFIRMED FALL ALERT] Time: {ts:.2f}s | "
                        f"Lead: +{lead_time:.2f}s | Peak ML: {c.get('peak_ml_probability', 0.0):.3f}",
                        flush=True,
                    )
                    p_info = get_patient_info(session_id)
                    try:
                        createFallAlert(
                            patientId=p_info["patient_id"],
                            roomId=p_info["room"],
                            timestamp=ts,
                            fall_details={
                                "session_id": session_id,
                                "source": "Intel RealSense D435 Live",
                                "lead_time": lead_time,
                                "peak_ml_probability": c.get("peak_ml_probability"),
                                "heuristic_confirmation_time": c.get("heuristic_confirmation_time", ts),
                            },
                        )
                    except Exception as err:
                        print(f"  [Dashboard Warning] Failed to dispatch alert: {err}", flush=True)

            # Rejected False Alarms
            if res.get("rejected_alarm"):
                r = res["rejected_alarm"]
                print(
                    f"  --- [REJECTED ALARM] Time: {ts:.2f}s | "
                    f"Peak ML: {r.get('peak_ml_probability', 0.0):.3f} | {r.get('rejection_reason')}",
                    flush=True,
                )

            # Rendering & Display
            if visualize or save_viz:
                disp = draw_detection_overlay(frame, res, fps_estimate=avg_recent_fps, is_live=True)

                if video_writer is not None:
                    video_writer.write(disp)
                elif save_viz and (not saved_snapshot or res.get("confirmed_fall")):
                    save_viz.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_viz), disp)
                    saved_snapshot = True

                if visualize:
                    cv2.imshow("Hybrid Fall Detector - Intel RealSense D435 Live", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[User Interrupt] 'q' key pressed. Stopping live stream safely...")
                        break

    except KeyboardInterrupt:
        print("\n[User Interrupt] KeyboardInterrupt received. Stopping...")

    finally:
        # Safe resource release
        if pipeline is not None:
            try:
                pipeline.stop()
                print("[RealSense] Hardware pipeline stopped and device unlocked.")
            except Exception:
                pass

        if cap_mock is not None:
            cap_mock.release()

        if video_writer is not None:
            video_writer.release()
            print(f"[Recording] Annotated video saved successfully ({frame_idx} frames).")

        if visualize:
            cv2.destroyAllWindows()

    total_wall_time = max(0.001, time.perf_counter() - t_start)
    overall_fps = frame_idx / total_wall_time
    confirmed_count = len([e for e in detector.event_log if e.get("status") == "CONFIRMED_FALL"])
    rejected_count = len([e for e in detector.event_log if e.get("status") == "REJECTED_ML_FALSE_ALARM"])

    print("\n" + "=" * 70)
    print("INTEL REALSENSE LIVE STREAM EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Frames Processed: {frame_idx}")
    print(f"  Total Duration:   {total_wall_time:.2f}s")
    print(f"  Observed Live FPS:{overall_fps:.2f} FPS")
    print(f"  Confirmed Falls:  {confirmed_count}")
    print(f"  Rejected Alarms:  {rejected_count}")
    print("=" * 70)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(detector.event_log, f, indent=2)
        print(f"Saved event log to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Live Intel RealSense D435 Fall Detection Runner (Production H5 Pipeline)"
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config" / "hybrid_detector_config.yaml", help="Path to config YAML")
    parser.add_argument("--fps", type=int, default=30, help="Camera target FPS (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense device serial number")
    parser.add_argument("--no-visualize", action="store_true", help="Run headless without OpenCV GUI window")
    parser.add_argument("--save-viz", type=Path, help="Save video or snapshot to path (.mp4, .jpg)")
    parser.add_argument("--output-json", type=Path, help="Path to save event log JSON")
    parser.add_argument("--max-frames", type=int, help="Optional limit on frames to process")
    parser.add_argument("--session-id", type=str, default="live_realsense", help="Patient/Session identifier")
    parser.add_argument("--mock-stream", type=Path, help="Simulate RealSense stream using video file for testing")

    args = parser.parse_args()

    detector = RealtimeHybridDetector(config_path=args.config)

    run_realsense_stream(
        detector=detector,
        width=args.width,
        height=args.height,
        fps=args.fps,
        visualize=not args.no_visualize,
        output_json=args.output_json,
        max_frames=args.max_frames,
        save_viz=args.save_viz,
        session_id=args.session_id,
        serial=args.serial,
        mock_source=args.mock_stream,
    )


if __name__ == "__main__":
    main()

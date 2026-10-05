#!/usr/bin/env python3
"""
test_realsense_live.py - Automated Verification Suite for Live Intel RealSense D435 Mode.
========================================================================================
Validates all required acceptance criteria:
  TEST A — Camera Startup & Hardware Interface
  TEST B — Normal Standing/Walking (0 falls, 0 dashboard alerts)
  TEST C — Multi-Person Visualization & Target Isolation (Bystanders unmonitored)
  TEST D — Controlled Genuine Fall -> Pipeline Confirmation -> Dashboard Alert
  TEST E — Nurse Dashboard Response & Escalation Invariance
  MODEL INTEGRITY — Invariance of frozen weights SHA-256
"""

import sys
import os
import time
import json
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import cv2

# Project root setup
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hybrid_detector import RealtimeHybridDetector
from run_realsense_live import (
    check_realsense_availability,
    init_realsense_pipeline,
    run_realsense_stream,
)
from run_hybrid_pipeline import draw_detection_overlay
from alert_manager import createFallAlert

EXPECTED_GRU_SHA256 = "38a5e9c316b04f9430c6930027cbac42d334eb8e4962845d712c6c9dc50a80d2"
EXPECTED_YOLO_SHA256 = "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"
SERVER_URL = "http://127.0.0.1:8000"


def is_server_running() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/state", timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def reset_dashboard():
    try:
        req = urllib.request.Request(f"{SERVER_URL}/api/alerts/reset", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[Warning] Failed to reset dashboard: {e}")
        return None


def get_dashboard_state() -> Dict[str, Any]:
    with urllib.request.urlopen(f"{SERVER_URL}/api/state", timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def respond_alert(alert_id: str, responder: str = "Nurse Priya"):
    payload = json.dumps({"responder_name": responder}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/alerts/{alert_id}/respond",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_model_invariance():
    print("\n[VERIFY] Model Integrity Check...")
    gru_path = BASE_DIR / "models" / "best_fall_gru.pt"
    yolo_path = BASE_DIR / "yolov8n-pose.pt"

    with open(gru_path, "rb") as f:
        gru_hash = hashlib.sha256(f.read()).hexdigest().lower()
    with open(yolo_path, "rb") as f:
        yolo_hash = hashlib.sha256(f.read()).hexdigest().lower()

    assert gru_hash == EXPECTED_GRU_SHA256, f"GRU hash mismatch! Got: {gru_hash}"
    assert yolo_hash == EXPECTED_YOLO_SHA256, f"YOLO hash mismatch! Got: {yolo_hash}"
    print(f"  --> PASS: best_fall_gru.pt SHA256 verified ({gru_hash[:16]}...)")
    print(f"  --> PASS: yolov8n-pose.pt   SHA256 verified ({yolo_hash[:16]}...)")


def test_a_camera_interface():
    print("\n" + "=" * 70)
    print("TEST A — REAL-SENSE CAMERA STARTUP & HARDWARE INTERFACE")
    print("=" * 70)

    avail, msg, devs = check_realsense_availability()
    print(f"  Availability check: {msg}")

    if not avail:
        print("  [Notice] Physical camera not connected during automated test run.")
        print("  Verifying graceful error handling without crashing...")
        pipe, prof, w, h, fps = init_realsense_pipeline()
        assert pipe is None, "Expected pipeline to be None when camera is missing"
        print("  --> PASS: Handled missing camera safely with clean warning and zero traceback.")
    else:
        print(f"  Found {len(devs)} RealSense device(s). Initializing pipeline...")
        pipe, prof, w, h, fps = init_realsense_pipeline(width=640, height=480, fps=30)
        assert pipe is not None, "Failed to initialize connected RealSense camera"
        assert w == 640 and h == 480, f"Expected 640x480, got {w}x{h}"
        assert fps in (15, 30, 60), f"Unexpected FPS: {fps}"

        # Capture sample frame
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            col = frames.get_color_frame()
            assert col is not None, "Failed to get color frame"
            data = np.asanyarray(col.get_data())
            assert data.shape == (480, 640, 3), f"Unexpected frame shape: {data.shape}"
            print(f"  --> PASS: Successfully captured live RGB frame: {data.shape} @ {fps} FPS")
        finally:
            pipe.stop()
            print("  --> PASS: RealSense pipeline stopped cleanly and camera released.")


def test_b_normal_movement():
    print("\n" + "=" * 70)
    print("TEST B — NORMAL STANDING & WALKING (0 FALLS, 0 DASHBOARD ALERTS)")
    print("=" * 70)

    normal_vid = BASE_DIR / "extracted" / "20260817_154129" / "preview.mp4"
    assert normal_vid.exists(), f"Sample video not found: {normal_vid}"

    reset_dashboard()
    detector = RealtimeHybridDetector()

    results = run_realsense_stream(
        detector=detector,
        mock_source=normal_vid,
        visualize=False,
        max_frames=120,  # 8 seconds
        session_id="test_normal_movement",
    )

    confirmed_falls = [r for r in results if r.get("confirmed_fall")]
    print(f"  Processed {len(results)} frames of normal movement.")
    print(f"  Confirmed falls detected: {len(confirmed_falls)}")
    assert len(confirmed_falls) == 0, f"Expected 0 confirmed falls, got {len(confirmed_falls)}"

    # Check person and keypoint detection
    frames_with_person = [r for r in results if r.get("has_person")]
    assert len(frames_with_person) > 0, "Expected person to be detected"
    first_person = frames_with_person[0]
    assert first_person.get("keypoints") is not None, "Expected valid keypoints"
    assert len(first_person["keypoints"]) == 17, "Expected 17 COCO keypoints"

    # Verify 0 dashboard alerts
    if is_server_running():
        st = get_dashboard_state()
        print(f"  Dashboard active alerts: {len(st['active_alerts'])}")
        assert len(st["active_alerts"]) == 0, "Dashboard must receive 0 alerts for normal movement"

    print("  --> PASS: Normal activity produced zero fall alerts on detector and dashboard.")


def test_c_multi_person_monitoring():
    print("\n" + "=" * 70)
    print("TEST C — MULTI-PERSON VISUALIZATION & TARGET ISOLATION")
    print("=" * 70)

    two_person_vid = BASE_DIR / "extracted" / "20260817_155334" / "preview.mp4"
    assert two_person_vid.exists(), f"Sample video not found: {two_person_vid}"

    detector = RealtimeHybridDetector()
    results = run_realsense_stream(
        detector=detector,
        mock_source=two_person_vid,
        visualize=False,
        max_frames=150,  # 10 seconds
        session_id="test_multi_person",
    )

    multi_person_frames = [r for r in results if len(r.get("all_boxes", [])) >= 2]
    print(f"  Total frames: {len(results)} | Multi-person frames: {len(multi_person_frames)}")
    assert len(multi_person_frames) > 0, "Expected multi-person frames"

    # Verify primary vs unmonitored tagging and keypoints
    sample = multi_person_frames[0]
    boxes = sample["all_boxes"]
    target_count = sum(1 for b in boxes if b.get("is_target"))
    unmonitored_count = sum(1 for b in boxes if not b.get("is_target"))

    print(f"  Frame has {target_count} target person and {unmonitored_count} unmonitored bystander(s).")
    assert target_count == 1, "Expected exactly 1 primary target locked"
    assert unmonitored_count >= 1, "Expected unmonitored secondary persons"

    # Verify keypoints present for both persons
    for b in boxes:
        kpts = b.get("keypoints")
        assert kpts is not None and len(kpts) == 17, "All persons must receive 17 COCO keypoints"
        assert "track_id" in b, "All persons must have a Track ID"

    # Verify zero false alarms triggered by secondary person
    confirmed_falls = [r for r in results if r.get("confirmed_fall")]
    print(f"  Confirmed falls: {len(confirmed_falls)}")
    assert len(confirmed_falls) == 0, "Unmonitored persons must never trigger fall alerts"

    # Test visualization overlay generation with LIVE badge
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = draw_detection_overlay(dummy_frame, sample, fps_estimate=25.0, is_live=True)
    assert overlay.shape == (480, 640, 3)

    print("  --> PASS: Multi-person pose visualization, track IDs, and bystander isolation verified.")


def test_d_controlled_fall_and_dashboard_alert():
    print("\n" + "=" * 70)
    print("TEST D — SIMULATED GENUINE FALL & DASHBOARD ALERT DISPATCH")
    print("=" * 70)

    fall_vid = BASE_DIR / "extracted" / "20260828_133415" / "preview.mp4"
    assert fall_vid.exists(), f"Sample fall video not found: {fall_vid}"

    if not is_server_running():
        print("  [ERROR] Nurse Dashboard Server is not running at http://127.0.0.1:8000")
        print("  Please launch: python scripts/nurse_dashboard_server.py")
        sys.exit(1)

    reset_dashboard()
    detector = RealtimeHybridDetector()

    # Process genuine fall recording
    results = run_realsense_stream(
        detector=detector,
        mock_source=fall_vid,
        visualize=False,
        max_frames=450,  # ~15 seconds, spans the entire fall event
        session_id="20260828_133415",
    )

    confirmed_falls = [r for r in results if r.get("confirmed_fall")]
    print(f"  Confirmed falls detected by pipeline: {len(confirmed_falls)}")
    assert len(confirmed_falls) == 1, f"Expected exactly 1 confirmed fall, got {len(confirmed_falls)}"

    fall_event = confirmed_falls[0]["confirmed_fall"]
    print(f"  Fall Alert Time: {fall_event['final_alert_time']:.2f}s | Lead Time: +{fall_event['advance_lead_time_seconds']:.2f}s")
    assert fall_event["advance_lead_time_seconds"] > 0, "Expected positive predictive advance lead time"

    # Verify dashboard received the alert via createFallAlert()
    st = get_dashboard_state()
    active_alerts = st.get("active_alerts", [])
    print(f"  Dashboard active alerts count: {len(active_alerts)}")
    assert len(active_alerts) == 1, f"Expected 1 alert on dashboard, got {len(active_alerts)}"

    alert = active_alerts[0]
    print(f"  Dashboard Alert ID: {alert['alert_id']}")
    print(f"  Patient: {alert['patient_id']} | Room: {alert['room_bed']}")
    print(f"  Assigned Nurse: {alert['assigned_person_name']} | State: {alert['state']}")
    print(f"  Timer Active: {alert['timer_active']} | Remaining: {alert['timer_seconds_remaining']}s")

    assert alert["patient_id"] == "Patient 104"
    assert alert["room_bed"] == "Ward A - Bed 12"
    assert alert["state"] == "WAITING_FOR_RESPONSE"
    assert alert["timer_active"] is True
    assert alert["assigned_person_name"] == "Nurse Priya"

    print("  --> PASS: Genuine fall triggered confirmed alert and dispatched to Nurse Dashboard.")
    return alert["alert_id"]


def test_e_dashboard_response(alert_id: str):
    print("\n" + "=" * 70)
    print("TEST E — NURSE DASHBOARD RESPONSE & ESCALATION STOP")
    print("=" * 70)

    # Nurse Priya responds to the alert
    print(f"  Nurse Priya responding to alert {alert_id}...")
    resp = respond_alert(alert_id, responder="Nurse Priya")
    assert resp.get("status") == "success"

    # Check updated dashboard state
    st = get_dashboard_state()
    active = st["active_alerts"]
    assert len(active) >= 1
    updated_alert = [a for a in active if a["alert_id"] == alert_id][0]

    print(f"  Updated State: {updated_alert['state']}")
    print(f"  Responding Nurse: {updated_alert['responding_nurse']}")
    print(f"  Timer Active: {updated_alert['timer_active']}")

    assert updated_alert["state"] in ("ACKNOWLEDGED", "CARE_IN_PROGRESS")
    assert updated_alert["responding_nurse"] == "Nurse Priya"
    assert updated_alert["timer_active"] is False, "Timer must immediately stop when responded"

    print("  --> PASS: Nurse response successfully acknowledged, countdown timer stopped, escalation halted.")


def main():
    print("=" * 80)
    print("INTEL REALSENSE LIVE INPUT MODE — AUTOMATED TEST SUITE")
    print("=" * 80)

    # 1. Model Integrity Check
    test_model_invariance()

    # 2. Test A: Camera Interface
    test_a_camera_interface()

    # 3. Test B: Normal Movement
    test_b_normal_movement()

    # 4. Test C: Multi-person Visualization
    test_c_multi_person_monitoring()

    # 5. Test D: Genuine Fall
    alert_id = test_d_controlled_fall_and_dashboard_alert()

    # 6. Test E: Dashboard Response
    test_e_dashboard_response(alert_id)

    # 7. Final Model Integrity Check
    test_model_invariance()

    print("\n" + "=" * 80)
    print("ALL ACCEPTANCE TESTS (TEST A -> E) PASSED SUCCESSFULLY (100%)!")
    print("=" * 80)


if __name__ == "__main__":
    main()

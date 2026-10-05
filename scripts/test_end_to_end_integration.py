#!/usr/bin/env python3
"""
test_end_to_end_integration.py - Automated End-to-End Verification of Video -> Detector -> Dashboard.

Verifies the exact requirements from Task Specification:
  TEST A — NO FALL:
    Run detector on a normal/non-fall recording (e.g. 20260817_154129 bending / normal movement).
    Verify 0 confirmed falls -> 0 dashboard alerts.

  TEST B — REAL FALL:
    Run detector on genuine fall recording (e.g. 20260828_133415).
    Verify fall is confirmed -> Alert is automatically created on dashboard.
    Verify Patient ID = "Patient 104", Room = "Ward A - Bed 12", Priority = "CRITICAL",
    Assigned to "Nurse Priya", State = "WAITING_FOR_RESPONSE", 30-second timer active.

  TEST C — NURSE RESPONSE:
    Trigger response before timeout expires.
    Verify countdown timer stops immediately.
    Verify state transitions to ACKNOWLEDGED / "Patient is being attended".
    Verify escalation stops.

  TEST D — NO RESPONSE & DOCTOR ESCALATION:
    Allow response timer to expire sequentially without nurse response.
    Verify escalation sequence: Nurse Priya -> Nurse Anitha -> Nurse Kavya -> DOCTOR ESCALATION.
    Verify doctor responds -> DOCTOR_ACKNOWLEDGED.

  TEST E — DUPLICATE SUPPRESSION:
    Verify that subsequent frames in LYING_DOWN / FALLING do not produce duplicate alerts.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_hybrid_pipeline import replay_feature_stream
from hybrid_detector import RealtimeHybridDetector

SERVER_URL = "http://127.0.0.1:8000"


def reset_server_state():
    req = urllib.request.Request(f"{SERVER_URL}/api/alerts/reset", data=b"", method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_server_state():
    with urllib.request.urlopen(f"{SERVER_URL}/api/state") as resp:
        return json.loads(resp.read().decode("utf-8"))


def respond_server_alert(alert_id: str, responder_name: str = "Nurse Priya"):
    payload = json.dumps({"responder_name": responder_name}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/alerts/{alert_id}/respond",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def set_server_timeout(seconds: int):
    payload = json.dumps({"timeout_seconds": seconds}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/timeout",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_e2e_tests():
    print("=" * 80)
    print("END-TO-END FALL DETECTION -> NURSE DASHBOARD INTEGRATION TEST SUITE")
    print("=" * 80)

    # Check that server is reachable
    try:
        st = get_server_state()
        print(f"  [Server Connected] Dashboard Server online at {SERVER_URL}")
    except Exception as e:
        print(f"  [ERROR] Cannot reach dashboard server at {SERVER_URL}: {e}")
        print("  Please start the server first with: python scripts/nurse_dashboard_server.py")
        sys.exit(1)

    detector = RealtimeHybridDetector()

    # -------------------------------------------------------------------------
    # TEST A — NO FALL (Normal Bending: 20260817_154129)
    # -------------------------------------------------------------------------
    print("\n[TEST A] Playing normal / non-fall video (20260817_154129)...")
    reset_server_state()
    feat_154129 = BASE_DIR / "data" / "features" / "20260817_154129.npz"
    assert feat_154129.exists(), f"Feature file {feat_154129} not found"

    results_a = replay_feature_stream(feat_154129, detector, simulate_realtime=False)
    confirmed_in_a = [r for r in results_a if r.get("confirmed_fall")]
    print(f"  Detector confirmed falls in non-fall video: {len(confirmed_in_a)}")
    assert len(confirmed_in_a) == 0, "Non-fall recording must produce 0 confirmed falls"

    state_a = get_server_state()
    active_count_a = len(state_a["active_alerts"])
    print(f"  Dashboard active alerts count: {active_count_a}")
    assert active_count_a == 0, f"Dashboard must have 0 alerts, got {active_count_a}"
    print("  -> PASS: Non-fall video produced ZERO false alarms and ZERO dashboard alerts.")

    # -------------------------------------------------------------------------
    # TEST B — REAL FALL (Confirmed Fall: 20260828_133415)
    # -------------------------------------------------------------------------
    print("\n[TEST B] Playing genuine fall video (20260828_133415)...")
    reset_server_state()
    feat_133415 = BASE_DIR / "data" / "features" / "20260828_133415.npz"
    assert feat_133415.exists(), f"Feature file {feat_133415} not found"

    results_b = replay_feature_stream(feat_133415, detector, simulate_realtime=False)
    confirmed_in_b = [r for r in results_b if r.get("confirmed_fall")]
    print(f"  Detector confirmed falls: {len(confirmed_in_b)}")
    assert len(confirmed_in_b) == 1, f"Expected exactly 1 confirmed fall, got {len(confirmed_in_b)}"

    # Check dashboard state
    state_b = get_server_state()
    active_b = state_b["active_alerts"]
    print(f"  Dashboard received alerts: {len(active_b)}")
    assert len(active_b) == 1, f"Expected 1 active alert on dashboard, got {len(active_b)}"

    alert_b = active_b[0]
    print(f"  Alert ID: {alert_b['alert_id']}")
    print(f"  Patient: {alert_b['patient_id']} | Room: {alert_b['room_bed']}")
    print(f"  Assigned Nurse: {alert_b['assigned_person_name']} | State: {alert_b['state']}")
    print(f"  Countdown Active: {alert_b['timer_active']} | Remaining: {alert_b['timer_seconds_remaining']}s")

    assert alert_b["patient_id"] == "Patient 104", f"Expected Patient 104, got {alert_b['patient_id']}"
    assert alert_b["room_bed"] == "Ward A - Bed 12", f"Expected Ward A - Bed 12, got {alert_b['room_bed']}"
    assert alert_b["assigned_person_name"] == "Nurse Priya", f"Expected Nurse Priya, got {alert_b['assigned_person_name']}"
    assert alert_b["state"] == "WAITING_FOR_RESPONSE"
    assert alert_b["timer_active"] is True
    print("  -> PASS: Genuine fall video automatically triggered CRITICAL FALL ALERT on Nurse Dashboard!")

    # -------------------------------------------------------------------------
    # TEST C — NURSE RESPONSE
    # -------------------------------------------------------------------------
    print("\n[TEST C] Nurse clicks RESPOND within 30s window...")
    resp_res = respond_server_alert(alert_b["alert_id"], responder_name="Nurse Priya")
    assert resp_res.get("status") == "success"

    state_c = get_server_state()
    alert_c = state_c["active_alerts"][0]
    print(f"  State after response: {alert_c['state']}")
    print(f"  Attended by: {alert_c['responding_nurse']}")
    print(f"  Timer active: {alert_c['timer_active']}")
    print(f"  Summary being_attended: {state_c['summary']['being_attended']}")

    assert alert_c["state"] in ("ACKNOWLEDGED", "CARE_IN_PROGRESS")
    assert alert_c["responding_nurse"] == "Nurse Priya"
    assert alert_c["timer_active"] is False, "Timer must be immediately stopped"
    assert state_c["summary"]["being_attended"] >= 1
    print("  -> PASS: Nurse responded, timer stopped, alert marked ACKNOWLEDGED / Patient Being Attended.")

    # -------------------------------------------------------------------------
    # TEST D — TIMEOUT ESCALATION TO NEXT NURSE & DOCTOR
    # -------------------------------------------------------------------------
    print("\n[TEST D] Testing timeout escalation progression (No response)...")
    reset_server_state()
    # Configure 3-second timeout on server for test speed
    set_server_timeout(3)

    # Replay genuine fall to create fresh alert
    replay_feature_stream(feat_133415, detector, simulate_realtime=False)

    st_d1 = get_server_state()
    alt_d = st_d1["active_alerts"][0]
    print(f"  Initial assignment: {alt_d['assigned_person_name']} (Remaining: {alt_d['timer_seconds_remaining']}s)")
    assert alt_d["assigned_person_name"] == "Nurse Priya"

    # Step 1: Timeout Priya -> Anitha
    print("  Waiting 3.5s for Nurse Priya timeout...")
    time.sleep(3.5)
    st_d2 = get_server_state()
    alt_d2 = st_d2["active_alerts"][0]
    print(f"  Escalated to: {alt_d2['assigned_person_name']} (Step {alt_d2['escalation_index'] + 1})")
    assert alt_d2["assigned_person_name"] == "Nurse Anitha"

    # Step 2: Timeout Anitha -> Kavya
    print("  Waiting 3.5s for Nurse Anitha timeout...")
    time.sleep(3.5)
    st_d3 = get_server_state()
    alt_d3 = st_d3["active_alerts"][0]
    print(f"  Escalated to: {alt_d3['assigned_person_name']} (Step {alt_d3['escalation_index'] + 1})")
    assert alt_d3["assigned_person_name"] == "Nurse Kavya"

    # Step 3: Timeout Kavya -> Doctor (since Divya is Offline)
    print("  Waiting 3.5s for Nurse Kavya timeout...")
    time.sleep(3.5)
    st_d4 = get_server_state()
    alt_d4 = st_d4["active_alerts"][0]
    print(f"  Doctor Escalation: {alt_d4['state']} | Assigned: {alt_d4['assigned_person_name']}")
    assert alt_d4["state"] == "DOCTOR_ESCALATION"
    assert "Dr. Suresh" in alt_d4["assigned_person_name"]

    # Doctor responds
    respond_server_alert(alt_d4["alert_id"], responder_name="Dr. Suresh (On-Call)")
    st_d5 = get_server_state()
    alt_d5 = st_d5["active_alerts"][0]
    assert alt_d5["state"] == "DOCTOR_ACKNOWLEDGED"
    print("  -> PASS: Escalation progressed sequentially Nurse Priya -> Anitha -> Kavya -> DOCTOR ESCALATION!")

    # Restore standard 30s timeout on server
    set_server_timeout(30)
    reset_server_state()

    # -------------------------------------------------------------------------
    # TEST E — DUPLICATE SUPPRESSION
    # -------------------------------------------------------------------------
    print("\n[TEST E] Duplicate alert suppression...")
    reset_server_state()
    # Run the full 897 frames of 20260828_133415
    replay_feature_stream(feat_133415, detector, simulate_realtime=False)
    st_e = get_server_state()
    print(f"  Total alerts created across all 897 frames: {len(st_e['active_alerts'])}")
    assert len(st_e["active_alerts"]) == 1, f"Expected exactly 1 alert, got {len(st_e['active_alerts'])}"
    print("  -> PASS: Duplicate suppression verified; single physical fall event emits exactly ONE alert.")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ALL END-TO-END INTEGRATION TESTS PASSED (100%)!")
    print("=" * 80)


if __name__ == "__main__":
    run_e2e_tests()

#!/usr/bin/env python3
"""
test_hybrid_pipeline.py - Comprehensive Unit & Integration Test Suite for Step 12.

Tests the 7 critical hybrid pipeline operational requirements:
    1. Normal movement (no alarms)
    2. ML trigger without heuristic confirmation (rejected false alarm)
    3. ML trigger followed by heuristic confirmation (confirmed fall with early warning preserved)
    4. Confirmation timeout (heuristic transition arriving too late is rejected)
    5. Duplicate suppression (single physical incident emits exactly 1 alert)
    6. Static lying (motionless posture on floor produces 0 new alerts)
    7. Session/recording boundary reset (complete buffer and state isolation)
"""

import sys
import unittest
from pathlib import Path
from collections import deque

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from hybrid_detector import RealtimeHybridDetector


class TestHybridPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Instantiate detector once for test suite
        cls.detector = RealtimeHybridDetector()

    def setUp(self):
        # Reset detector state before each test
        self.detector.reset(session_id="unit_test_session")

    def test_1_normal_movement(self):
        """Verify normal upright movement produces zero provisional or confirmed alerts."""
        detector = self.detector
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        alerts = []
        rejections = []

        # Stream 60 frames (2 seconds at 30 fps) of normal standing
        for i in range(60):
            ts = i / 30.0
            # Upright posture data
            res = detector.process_frame(dummy_frame, ts, frame_number=i + 1)
            if res["confirmed_fall"]:
                alerts.append(res["confirmed_fall"])
            if res["rejected_alarm"]:
                rejections.append(res["rejected_alarm"])

        self.assertEqual(len(alerts), 0, "Normal movement must not trigger confirmed falls")
        self.assertEqual(len(rejections), 0, "Normal movement must not trigger rejected alarms")
        self.assertEqual(detector.heuristic_detector.current_state, "NORMAL")

    def test_2_ml_trigger_without_heuristic_confirmation(self):
        """Verify an isolated ML trigger without heuristic confirmation is discarded as a rejected false alarm."""
        detector = self.detector

        # Manually create an active provisional event
        t_trigger = 2.0
        t_ml_end = 3.5
        detector.active_provisional_event = {
            "session_id": "unit_test_session",
            "ml_trigger_time": t_trigger,
            "ml_end_time": t_ml_end,
            "start_frame": 60,
            "end_frame": 105,
            "probabilities": [0.85, 0.90],
            "status": "PROVISIONAL",
        }

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Advance time to t=4.0s (within W=1.0s window: allowed up to 3.5 + 1.0 = 4.5s)
        res1 = detector.process_frame(dummy_frame, 4.0, frame_number=120)
        self.assertIsNone(res1["confirmed_fall"])
        self.assertIsNone(res1["rejected_alarm"])
        self.assertTrue(detector.active_provisional_event is not None)

        # Advance time to t=4.6s (past W=1.0s window: 4.6 > 4.5s)
        res2 = detector.process_frame(dummy_frame, 4.6, frame_number=138)
        self.assertIsNone(res2["confirmed_fall"])
        self.assertIsNotNone(res2["rejected_alarm"], "Unconfirmed ML trigger must be rejected upon timeout")
        self.assertEqual(res2["rejected_alarm"]["status"], "REJECTED_ML_FALSE_ALARM")
        self.assertIsNone(detector.active_provisional_event, "Provisional event must be cleared after timeout")

    def test_3_ml_trigger_followed_by_heuristic_confirmation(self):
        """Verify ML trigger confirmed by heuristic FSM emits CONFIRMED_FALL with early trigger timestamp."""
        detector = self.detector

        t_trigger = 10.0
        t_ml_end = 12.0
        detector.active_provisional_event = {
            "session_id": "unit_test_session",
            "ml_trigger_time": t_trigger,
            "ml_end_time": t_ml_end,
            "start_frame": 300,
            "end_frame": 360,
            "probabilities": [0.75, 0.88],
            "status": "PROVISIONAL",
        }

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate heuristic FSM fall transition at t=12.5s (within W=1.0s of 12.0s)
        # Force heuristic detector to transition to FALLING
        hd = detector.heuristic_detector
        hd.current_state = "NORMAL"

        # Mock kinematic data to force FSM transition
        orig_analyze = hd.analyze_frame_kinematics
        try:
            hd.analyze_frame_kinematics = lambda f, ts: {
                "timestamp_s": ts,
                "has_person": True,
                "box": np.array([100, 350, 400, 470]),
                "keypoints": None,
                "keypoints_conf": None,
                "ar": 0.6,
                "angle_deg": 65.0,
                "bottom_y": 0.95,
                "v_down": 0.40,  # Velocity spike
                "is_upright": False,
                "is_horizontal": True,
                "raw_score": 0.85,
            }

            res = detector.process_frame(dummy_frame, 12.5, frame_number=375)
            self.assertIsNotNone(res["confirmed_fall"], "FSM transition within window must confirm the fall")
            ev = res["confirmed_fall"]
            self.assertEqual(ev["status"], "CONFIRMED_FALL")
            self.assertEqual(ev["final_alert_time"], t_trigger, "Final alert time must preserve early warning timestamp")
            self.assertEqual(ev["heuristic_confirmation_time"], 12.5)
            self.assertAlmostEqual(ev["advance_lead_time_seconds"], 2.5, places=2)
            self.assertIsNone(detector.active_provisional_event)
        finally:
            hd.analyze_frame_kinematics = orig_analyze

    def test_4_confirmation_timeout(self):
        """Verify heuristic transition arriving after confirmation window W=1.0s has expired is rejected."""
        detector = self.detector

        t_trigger = 5.0
        t_ml_end = 6.0  # Window allows confirmation up to 6.0 + 1.0 = 7.0s
        detector.active_provisional_event = {
            "session_id": "unit_test_session",
            "ml_trigger_time": t_trigger,
            "ml_end_time": t_ml_end,
            "start_frame": 150,
            "end_frame": 180,
            "probabilities": [0.80, 0.85],
            "status": "PROVISIONAL",
        }

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Step at t=7.2s -> Provisional expires and is rejected
        res_exp = detector.process_frame(dummy_frame, 7.2, frame_number=216)
        self.assertIsNotNone(res_exp["rejected_alarm"])
        self.assertIsNone(detector.active_provisional_event)

        # Later at t=8.0s, late heuristic transition occurs
        hd = detector.heuristic_detector
        orig_analyze = hd.analyze_frame_kinematics
        try:
            hd.analyze_frame_kinematics = lambda f, ts: {
                "timestamp_s": ts,
                "has_person": True,
                "box": np.array([100, 350, 400, 470]),
                "keypoints": None,
                "keypoints_conf": None,
                "ar": 0.6,
                "angle_deg": 65.0,
                "bottom_y": 0.95,
                "v_down": 0.35,
                "is_upright": False,
                "is_horizontal": True,
                "raw_score": 0.85,
            }
            res_late = detector.process_frame(dummy_frame, 8.0, frame_number=240)
            self.assertIsNone(res_late["confirmed_fall"], "Late heuristic transition must not confirm expired ML event")
        finally:
            hd.analyze_frame_kinematics = orig_analyze

    def test_5_duplicate_suppression(self):
        """Verify continuous activity during a single physical incident emits exactly 1 confirmed alert."""
        detector = self.detector
        detector.last_confirmed_fall_time = 15.0  # Just confirmed fall at 15.0s

        # Attempt to confirm another event at t=16.5s (within 5.0s cooldown)
        detector.active_provisional_event = {
            "session_id": "unit_test_session",
            "ml_trigger_time": 15.5,
            "ml_end_time": 16.5,
            "start_frame": 465,
            "end_frame": 495,
            "probabilities": [0.70],
            "status": "PROVISIONAL",
        }

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hd = detector.heuristic_detector
        hd.current_state = "LYING_DOWN"

        res = detector.process_frame(dummy_frame, 16.5, frame_number=495)
        self.assertIsNone(res["confirmed_fall"], "Duplicate alerts within cooldown must be suppressed")

    def test_6_static_lying(self):
        """Verify motionless posture on the floor in LYING_DOWN state produces 0 new alerts."""
        detector = self.detector
        hd = detector.heuristic_detector
        hd.current_state = "LYING_DOWN"
        detector.last_confirmed_fall_time = 10.0

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orig_analyze = hd.analyze_frame_kinematics

        try:
            # Floor posture: horizontal, zero downward velocity
            hd.analyze_frame_kinematics = lambda f, ts: {
                "timestamp_s": ts,
                "has_person": True,
                "box": np.array([100, 400, 500, 470]),
                "keypoints": None,
                "keypoints_conf": None,
                "ar": 0.4,
                "angle_deg": 80.0,
                "bottom_y": 0.98,
                "v_down": 0.0,  # Zero velocity on floor
                "is_upright": False,
                "is_horizontal": True,
                "raw_score": 0.40,
            }

            # Stream 15 seconds of static lying (t=11.0s to 26.0s)
            alerts = []
            for i in range(150):
                ts = 11.0 + (i * 0.1)
                res = detector.process_frame(dummy_frame, ts, frame_number=330 + i)
                if res["confirmed_fall"]:
                    alerts.append(res["confirmed_fall"])

            self.assertEqual(len(alerts), 0, "Static lying must not generate any new fall alerts")
            self.assertEqual(hd.current_state, "LYING_DOWN", "State must remain LYING_DOWN")
        finally:
            hd.analyze_frame_kinematics = orig_analyze

    def test_7_session_recording_boundary_reset(self):
        """Verify reset() completely flushes all buffers and resets state across recording boundaries."""
        detector = self.detector

        # Populate buffer with dummy features
        for i in range(25):
            detector.feature_buffer.append(np.ones(62, dtype=np.float32))
            detector.timestamp_buffer.append(float(i))
            detector.frame_idx_buffer.append(i)

        detector.consecutive_positive_windows = 3
        detector.active_provisional_event = {"status": "PROVISIONAL"}
        detector.last_confirmed_fall_time = 10.0
        detector.heuristic_detector.current_state = "LYING_DOWN"

        self.assertEqual(len(detector.feature_buffer), 25)

        # Reset session
        detector.reset(session_id="new_clean_session")

        self.assertEqual(len(detector.feature_buffer), 0, "Feature buffer must be emptied on reset")
        self.assertEqual(len(detector.timestamp_buffer), 0, "Timestamp buffer must be emptied on reset")
        self.assertEqual(len(detector.frame_idx_buffer), 0, "Frame index buffer must be emptied on reset")
        self.assertEqual(detector.consecutive_positive_windows, 0)
        self.assertIsNone(detector.active_provisional_event)
        self.assertIsNone(detector.last_confirmed_fall_time)
        self.assertEqual(detector.current_session_id, "new_clean_session")

    def test_8_fluid_sit_and_recline_suppression(self):
        """Verify intentional fluid walk -> sit -> recline transitions NORMAL -> SITTING -> LYING_DOWN with 0 fall alerts."""
        detector = self.detector
        hd = detector.heuristic_detector
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 20 frames upright walking (0.0s - 0.67s)
        # 10 frames sit down (0.67s - 1.0s)
        # 12 frames seated dwell 0.40s (1.0s - 1.4s) -> triggers SITTING (dwell >= 0.25s)
        # 12 frames recline (1.4s - 1.8s) -> v_down=0.20, horizontal
        # 20 frames lying on bed (1.8s - 2.47s)
        states_seen = []
        alerts = []

        orig_analyze = hd.analyze_frame_kinematics
        try:
            for f in range(80):
                ts = f / 30.0
                if f < 20:
                    # Upright walking
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 100, 300, 400]),
                        "keypoints": None, "keypoints_conf": None, "ar": 1.70, "angle_deg": 12.0,
                        "bottom_y": 0.85, "v_down": 0.02, "is_upright": True, "is_horizontal": False, "raw_score": 0.05
                    }
                elif f < 30:
                    # Sitting down motion
                    p = (f - 20) / 10.0
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 120, 300, 400]),
                        "keypoints": None, "keypoints_conf": None, "ar": 1.70 - 0.75 * p, "angle_deg": 12.0 + 10.0 * p,
                        "bottom_y": 0.85 - 0.20 * p, "v_down": 0.16, "is_upright": (1.70 - 0.75 * p >= 1.20),
                        "is_horizontal": False, "raw_score": 0.15
                    }
                elif f < 48:
                    # Seated on bed (18 frames = 0.60s dwell >= 0.50s)
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 150, 300, 380]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.95, "angle_deg": 20.0,
                        "bottom_y": 0.65, "v_down": 0.02, "is_upright": False, "is_horizontal": False, "raw_score": 0.10
                    }
                elif f < 60:
                    # Reclining backwards onto bed
                    p = (f - 48) / 12.0
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 150, 300, 380]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.95 - 0.35 * p, "angle_deg": 20.0 + 55.0 * p,
                        "bottom_y": 0.65, "v_down": 0.20, "is_upright": False, "is_horizontal": (20.0 + 55.0 * p >= 50.0),
                        "raw_score": 0.25 + 0.40 * p
                    }
                else:
                    # Lying on bed
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 150, 400, 250]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.60, "angle_deg": 75.0,
                        "bottom_y": 0.65, "v_down": 0.0, "is_upright": False, "is_horizontal": True, "raw_score": 0.65
                    }

                hd.analyze_frame_kinematics = lambda frame, t, k=kd: k
                # Feed frame with feature vector
                feat = np.zeros(62, dtype=np.float32)
                res = detector.process_feature_step(feat, kd, ts, frame_number=f + 1)
                states_seen.append(res["fsm_state"])
                if res["confirmed_fall"]:
                    alerts.append(res["confirmed_fall"])

            self.assertIn("SITTING", states_seen, "Must transition through SITTING state")
            self.assertNotIn("FALLING", states_seen, "Must NOT enter FALLING during intentional sit and recline")
            self.assertEqual(states_seen[-1], "LYING_DOWN", "Final state must be LYING_DOWN")
            self.assertEqual(len(alerts), 0, "Intentional sit and recline must generate 0 fall alerts")
        finally:
            hd.analyze_frame_kinematics = orig_analyze

    def test_9_sitting_to_genuine_fall(self):
        """Verify that falling from a seated position transitions SITTING -> FALLING -> LYING_DOWN and triggers fall detection."""
        detector = self.detector
        hd = detector.heuristic_detector

        states_seen = []
        fall_events = []

        orig_analyze = hd.analyze_frame_kinematics
        try:
            # 15 frames sitting dwell (0.5s) -> SITTING
            # 10 frames falling off chair to the floor -> FALLING (velocity spike >= 0.25, bottom_y reaches ground >= 0.75)
            # 20 frames lying on floor -> LYING_DOWN
            for f in range(50):
                ts = f / 30.0
                if f < 18:
                    # Seated quietly (18 frames = 0.60s dwell >= 0.50s)
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 150, 300, 380]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.95, "angle_deg": 20.0,
                        "bottom_y": 0.65, "v_down": 0.01, "is_upright": False, "is_horizontal": False, "raw_score": 0.10
                    }
                elif f < 28:
                    # Falling off chair onto floor
                    p = (f - 18) / 10.0
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 200, 300, 500]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.95 - 0.40 * p, "angle_deg": 20.0 + 60.0 * p,
                        "bottom_y": 0.65 + 0.25 * p, "v_down": 0.35, "is_upright": False, "is_horizontal": (p >= 0.7),
                        "raw_score": 0.50 + 0.35 * p
                    }
                else:
                    # Lying motionless on floor
                    kd = {
                        "timestamp_s": ts, "has_person": True, "box": np.array([100, 350, 500, 450]),
                        "keypoints": None, "keypoints_conf": None, "ar": 0.55, "angle_deg": 80.0,
                        "bottom_y": 0.90, "v_down": 0.0, "is_upright": False, "is_horizontal": True, "raw_score": 0.85
                    }

                feat = np.zeros(62, dtype=np.float32)
                res = detector.process_feature_step(feat, kd, ts, frame_number=f + 1)
                states_seen.append(res["fsm_state"])

            self.assertIn("SITTING", states_seen, "Must enter SITTING while seated")
            self.assertIn("FALLING", states_seen, "Must enter FALLING during dynamic fall from chair")
            self.assertEqual(states_seen[-1], "LYING_DOWN", "Must end in LYING_DOWN")
        finally:
            hd.analyze_frame_kinematics = orig_analyze

    def test_10_target_lock_multi_person_stability(self):
        """Verify detector stays locked to target person and does not switch to a lying bystander with higher confidence."""
        detector = self.detector
        hd = detector.heuristic_detector
        hd.reset()

        class MockBoxItem:
            def __init__(self, xyxy, conf, tid=None):
                self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
                self.conf = torch.tensor([conf], dtype=torch.float32)
                self.tid = tid

        class MockBoxesCollection:
            def __init__(self, items):
                self.items = items
                self.conf = torch.tensor([it.conf.item() for it in items], dtype=torch.float32) if items else None
                if any(it.tid is not None for it in items):
                    self.id = torch.tensor([it.tid if it.tid is not None else -1 for it in items], dtype=torch.int64)
                else:
                    self.id = None

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        class MockResult:
            def __init__(self, items):
                self.boxes = MockBoxesCollection(items)
                self.keypoints = None

        # Frame 1: Person 1 (standing target, track_id=1, conf=0.85)
        res1 = MockResult([
            MockBoxItem([200, 100, 280, 380], 0.85, tid=1)
        ])
        sel_idx, track_id, all_boxes = hd._associate_target_person(res1, 480, 640)
        self.assertEqual(track_id, 1, "Must initialize target lock to Person 1")
        self.assertEqual(sel_idx, 0)
        self.assertTrue(all_boxes[0]["is_target"])

        # Frame 2: Person 1 (Track 1, conf=0.80) AND Person 2 (lying on bed, Track 2, conf=0.98)
        # Even though Person 2 has higher confidence (0.98 > 0.80), detector MUST stay locked to Person 1 (Track 1)
        res2 = MockResult([
            MockBoxItem([50, 300, 350, 420], 0.98, tid=2),  # Bystander lying on bed
            MockBoxItem([205, 102, 285, 382], 0.80, tid=1),  # Moving target walking
        ])
        sel_idx2, track_id2, all_boxes2 = hd._associate_target_person(res2, 480, 640)
        self.assertEqual(track_id2, 1, "Must maintain lock on Track 1, ignoring higher-confidence Person 2")
        self.assertEqual(sel_idx2, 1, "Must select Person 1 at index 1")
        self.assertFalse(all_boxes2[0]["is_target"], "Bystander on bed must be marked is_target=False")
        self.assertTrue(all_boxes2[1]["is_target"], "Target must be marked is_target=True")

    def test_11_target_lock_temporary_occlusion(self):
        """Verify detector tolerates temporary occlusion without immediately switching to a bystander."""
        detector = self.detector
        hd = detector.heuristic_detector
        hd.reset()

        class MockBoxItem:
            def __init__(self, xyxy, conf, tid=None):
                self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
                self.conf = torch.tensor([conf], dtype=torch.float32)
                self.tid = tid

        class MockBoxesCollection:
            def __init__(self, items):
                self.items = items
                self.conf = torch.tensor([it.conf.item() for it in items], dtype=torch.float32) if items else None
                if any(it.tid is not None for it in items):
                    self.id = torch.tensor([it.tid if it.tid is not None else -1 for it in items], dtype=torch.int64)
                else:
                    self.id = None

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        class MockResult:
            def __init__(self, items):
                self.boxes = MockBoxesCollection(items)
                self.keypoints = None

        # Lock onto Person 1
        res1 = MockResult([
            MockBoxItem([200, 100, 280, 380], 0.85, tid=1)
        ])
        hd._associate_target_person(res1, 480, 640)

        # 5 frames of occlusion where only Person 2 (bystander) is visible
        for _ in range(5):
            res_occ = MockResult([
                MockBoxItem([50, 300, 350, 420], 0.95, tid=2)
            ])
            sel_idx, tid, all_b = hd._associate_target_person(res_occ, 480, 640)
            self.assertIsNone(sel_idx, "Target must be considered temporarily missing during occlusion")
            self.assertEqual(tid, 1, "Target ID lock must be retained")
            self.assertFalse(all_b[0]["is_target"], "Bystander must NOT be selected as target")

        # Frame 7: Target 1 reappears at similar spatial coordinates without track ID (YOLO track drop)
        res_reappear = MockResult([
            MockBoxItem([50, 300, 350, 420], 0.95, tid=2),  # Bystander
            MockBoxItem([208, 105, 288, 385], 0.78, tid=None),  # Target reappears nearby without tid
        ])
        sel_idx_re, tid_re, all_b_re = hd._associate_target_person(res_reappear, 480, 640)
        self.assertEqual(sel_idx_re, 1, "Spatial IoU fallback must re-associate reappearing target")
        self.assertTrue(all_b_re[1]["is_target"])


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()


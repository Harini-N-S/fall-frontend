#!/usr/bin/env python3
"""
hybrid_detector.py - Real-Time Hybrid Fall Detection Pipeline (Step 12).

Implements Strategy H5:
    ML Early Predictive Trigger + Refined Heuristic FSM Confirmation (W = 1.0s)

Key Architecture:
    1. Input RGB Frame -> YOLOv8-Pose (640x480)
    2. Pose & Kinematic Analysis -> Refined FSM (NORMAL -> FALLING -> LYING_DOWN)
    3. 62-d Normalized Feature Extraction -> Rolling 32-Frame Buffer (stride=8)
    4. Frozen FallGRUClassifier (Threshold=0.5, M=2, G=0, D=0) -> Provisional Trigger
    5. Hybrid Confirmation Gate:
       - Confirms if Heuristic FSM verifies fall within [t_trigger, t_ml_end + 1.0s]
       - Emits CONFIRMED_FALL preserving ML early warning timestamp
       - Rejects unconfirmed provisional triggers upon timeout
       - Suppresses duplicate alerts and static lying false alarms
"""

import sys
import os
import yaml
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import torch

# Add scripts directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from model import FallGRUClassifier
from fall_detector_refined import RefinedFallDetector, COCO_SKELETON


class RealtimeHybridDetector:
    """
    Streaming Hybrid Fall Detector executing Strategy H5 in real time.
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = BASE_DIR / "config" / "hybrid_detector_config.yaml"
            if not config_path.exists():
                config_path = BASE_DIR / "config" / "detector_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 1. Kinematics & Heuristic Detector Setup
        self.heuristic_detector = RefinedFallDetector(config_path)
        k_cfg = self.cfg.get("kinematics", {})
        self.min_fall_vel: float = float(k_cfg.get("min_fall_velocity", 0.15))
        self.early_angle_thresh: float = float(k_cfg.get("early_angle_threshold", 22.0))
        self.early_ar_thresh: float = float(k_cfg.get("early_ar_threshold", 1.25))
        self.sitting_ar_min: float = float(k_cfg.get("sitting_aspect_ratio_min", 0.70))
        self.sitting_ar_max: float = float(k_cfg.get("sitting_aspect_ratio_max", 1.05))
        self.sitting_angle_max: float = float(k_cfg.get("sitting_torso_angle_max", 35.0))
        self.sitting_knee_angle_max: float = float(k_cfg.get("sitting_knee_angle_max", 130.0))
        self.sitting_vel_max: float = float(k_cfg.get("sitting_velocity_max", 0.12))
        self.sitting_min_dwell_s: float = float(k_cfg.get("sitting_dwell_seconds", 0.50))
        self.sitting_min_bottom_y: float = float(k_cfg.get("sitting_min_bottom_y", 0.40))
        self.bending_angle_min: float = float(k_cfg.get("bending_torso_angle_min", 35.0))
        self.bending_knee_angle_min: float = float(k_cfg.get("bending_knee_angle_min", 135.0))
        self.bending_ar_min: float = float(k_cfg.get("bending_aspect_ratio_min", 0.85))

        # 2. ML Pipeline Setup
        ml_cfg = self.cfg.get("ml_pipeline", {})
        self.temporal_window: int = ml_cfg.get("temporal_window", 32)
        self.feature_dim: int = ml_cfg.get("feature_dim", 62)
        self.stride: int = ml_cfg.get("stride", 8)
        self.ml_threshold: float = ml_cfg.get("ml_threshold", 0.50)
        self.postprocess_M: int = ml_cfg.get("postprocess_M", 2)
        self.postprocess_G: int = ml_cfg.get("postprocess_G", 0)
        self.postprocess_D: float = ml_cfg.get("postprocess_D", 0.0)

        # 3. Hybrid Decision Setup
        hy_cfg = self.cfg.get("hybrid_decision", {})
        self.strategy: str = hy_cfg.get("strategy", "H5")
        self.confirmation_window_s: float = hy_cfg.get("confirmation_window_seconds", 1.0)
        self.cooldown_s: float = hy_cfg.get("duplicate_suppression_cooldown_seconds", 5.0)
        self.suppress_during_static_lying: bool = bool(hy_cfg.get("suppress_during_static_lying", True))

        # 4. Model Loading
        model_cfg = self.cfg.get("model", {})
        ckpt_path_rel = model_cfg.get("gru_checkpoint", "models/best_fall_gru.pt")
        ckpt_path = (BASE_DIR / ckpt_path_rel).resolve()
        self.device = torch.device(model_cfg.get("device", "cpu"))

        self.gru_model = self._load_frozen_gru(ckpt_path)

        # 5. Session State
        self.current_session_id: str = "default_session"
        self.reset(session_id=self.current_session_id)

    def _load_frozen_gru(self, ckpt_path: Path) -> FallGRUClassifier:
        """Loads and freezes FallGRUClassifier on CPU."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GRU Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        m_cfg = checkpoint.get("model_config", {})
        input_size = m_cfg.get("input_size", self.feature_dim)
        hidden_size = m_cfg.get("hidden_size", 128)
        num_layers = m_cfg.get("num_layers", 2)
        dropout = m_cfg.get("dropout", 0.2)

        model = FallGRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        return model

    def reset(self, session_id: str = "default_session"):
        """
        Resets all internal rolling buffers and states on recording/session boundary.
        Enforces strict boundary isolation (Requirement 4).
        """
        self.current_session_id = session_id
        self.heuristic_detector.reset()

        # Rolling feature buffers for ML
        self.feature_buffer: deque = deque(maxlen=self.temporal_window)
        self.timestamp_buffer: deque = deque(maxlen=self.temporal_window)
        self.frame_idx_buffer: deque = deque(maxlen=self.temporal_window)

        # ML consecutive tracking
        self.consecutive_positive_windows: int = 0
        self.positive_window_history: List[Dict[str, Any]] = []

        # Provisional event tracking
        self.provisional_events: List[Dict[str, Any]] = []
        self.current_ml_run: Optional[Dict[str, Any]] = None

        # Hybrid confirmed tracking
        self.last_confirmed_fall_time: Optional[float] = None
        self.last_confirmed_heuristic_time: Optional[float] = None

        # Frame counter
        self.session_frame_count: int = 0
        self.last_valid_norm_box = np.zeros(4, dtype=np.float32)
        self.last_valid_kpts_norm = np.zeros((17, 2), dtype=np.float32)

        # Logged event history
        self.event_log: List[Dict[str, Any]] = []
        self.recent_heuristic_falls: deque = deque(maxlen=30)
        self.downward_motion_frames: int = 0
        self.recent_early_transitions: deque = deque(maxlen=30)

    @property
    def active_provisional_event(self) -> Optional[Dict[str, Any]]:
        return self.provisional_events[0] if self.provisional_events else None

    @active_provisional_event.setter
    def active_provisional_event(self, val: Optional[Dict[str, Any]]):
        if val is None:
            self.provisional_events.clear()
            self.current_ml_run = None
        else:
            self.provisional_events = [val]
            self.current_ml_run = val

    def extract_62_features(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts the exact 62-d feature vector per frame matching extract_features.py.
        """
        h_img, w_img = frame.shape[:2]
        has_person = bool(data.get("has_person", False))

        # Normalized coordinates extraction
        if has_person and data.get("box") is not None:
            box = data["box"]  # [x1, y1, x2, y2]
            box_norm = np.array([
                box[0] / max(1.0, w_img),
                box[1] / max(1.0, h_img),
                box[2] / max(1.0, w_img),
                box[3] / max(1.0, h_img),
            ], dtype=np.float32)
            box_w_norm = float(box_norm[2] - box_norm[0])
            box_h_norm = float(box_norm[3] - box_norm[1])
            self.last_valid_norm_box = box_norm.copy()
        else:
            box_norm = self.last_valid_norm_box.copy()
            box_w_norm = float(box_norm[2] - box_norm[0])
            box_h_norm = float(box_norm[3] - box_norm[1])

        # Keypoints & confidence
        if has_person and data.get("keypoints") is not None:
            raw_kpts = data["keypoints"]  # (17, 2)
            kpts_norm = np.zeros((17, 2), dtype=np.float32)
            kpts_norm[:, 0] = raw_kpts[:, 0] / max(1.0, w_img)
            kpts_norm[:, 1] = raw_kpts[:, 1] / max(1.0, h_img)
            confs = (
                data["keypoints_conf"].astype(np.float32)
                if data.get("keypoints_conf") is not None
                else np.ones(17, dtype=np.float32)
            )
            self.last_valid_kpts_norm = kpts_norm.copy()
        else:
            kpts_norm = self.last_valid_kpts_norm.copy()
            confs = np.zeros(17, dtype=np.float32)

        ar = float(data.get("ar", 1.5))
        angle_deg = float(data.get("angle_deg", 10.0))
        v_down = float(data.get("v_down", 0.0))
        bottom_y = float(data.get("bottom_y", 0.5))
        has_person_val = 1.0 if has_person else 0.0

        # Construct 62-d vector
        feat = np.zeros(62, dtype=np.float32)
        # 0..33: 34 flattened keypoints (x, y)
        feat[0:34] = kpts_norm.reshape(34)
        # 34..50: 17 confidences
        feat[34:51] = confs
        # 51..54: 4 box coords
        feat[51:55] = box_norm
        # 55..56: box dimensions
        feat[55] = box_w_norm
        feat[56] = box_h_norm
        # 57..61: kinematics
        feat[57] = ar
        feat[58] = angle_deg
        feat[59] = v_down
        feat[60] = bottom_y
        feat[61] = has_person_val

        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

    def kinematics_from_array(
        self,
        kin: np.ndarray,
        ts: float,
        kpts: Optional[np.ndarray] = None,
        kconfs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Reconstructs kinematics dict from saved feature array:
        kin = [box_w, box_h, ar, angle_deg, v_down, bottom_y, has_person]
        """
        ar = float(kin[2])
        angle_deg = float(kin[3])
        v_down = float(kin[4])
        bottom_y = float(kin[5])
        has_person = bool(kin[6] > 0.5)

        is_upright = (ar >= 1.20) and (angle_deg <= 30.0)

        # Check legs extended / supported by legs if keypoints available
        legs_extended = False
        if kpts is not None and kconfs is not None:
            knee_angles = []
            if (len(kpts) > 15 and len(kconfs) > 15
                and kconfs[11] > 0.30 and kconfs[13] > 0.30 and kconfs[15] > 0.30):
                knee_angles.append(self.heuristic_detector._compute_joint_angle(kpts[11], kpts[13], kpts[15]))
            if (len(kpts) > 16 and len(kconfs) > 16
                and kconfs[12] > 0.30 and kconfs[14] > 0.30 and kconfs[16] > 0.30):
                knee_angles.append(self.heuristic_detector._compute_joint_angle(kpts[12], kpts[14], kpts[16]))
            if len(knee_angles) > 0:
                avg_knee = float(np.mean(knee_angles))
                legs_extended = (avg_knee >= self.bending_knee_angle_min)

        is_bending = (
            (angle_deg >= self.bending_angle_min)
            and (legs_extended or (ar >= 1.05 and bottom_y >= 0.40))
            and (ar >= self.bending_ar_min)
        )

        # Horizontal posture on bed or floor (gated to exclude standing leg-supported bend)
        is_horizontal = ((ar <= 0.90) or (angle_deg >= 50.0)) and (bottom_y >= 0.35) and not is_bending

        s_ar = float(np.clip((1.20 - ar) / (1.20 - 0.90), 0.0, 1.0))
        s_angle = float(np.clip((angle_deg - 30.0) / (50.0 - 30.0), 0.0, 1.0))
        s_floor = float(np.clip((bottom_y - 0.50) / (0.70 - 0.50), 0.0, 1.0))
        s_vel = float(np.clip(v_down / 0.25, 0.0, 1.0))
        raw_score = (0.40 * s_angle + 0.30 * s_ar + 0.15 * s_floor + 0.15 * s_vel) if has_person else 0.0

        return {
            "timestamp_s": ts,
            "has_person": has_person,
            "box": None,
            "keypoints": kpts,
            "keypoints_conf": kconfs,
            "ar": ar,
            "angle_deg": angle_deg,
            "v_down": v_down,
            "bottom_y": bottom_y,
            "is_upright": is_upright,
            "is_horizontal": is_horizontal,
            "is_bending": is_bending,
            "raw_score": float(np.clip(raw_score, 0.0, 1.0)),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        frame_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single incoming RGB frame in real time.
        """
        data = self.heuristic_detector.analyze_frame_kinematics(frame, timestamp_s)
        feat_vector = self.extract_62_features(frame, data)
        return self.process_feature_step(feat_vector, data, timestamp_s, frame_number)

    def process_feature_step(
        self,
        feat_vector: np.ndarray,
        data: Dict[str, Any],
        timestamp_s: float,
        frame_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single pre-extracted feature vector and kinematic dict in real time.
        """
        self.session_frame_count += 1
        f_idx = frame_number if frame_number is not None else self.session_frame_count

        # 1. Update rolling buffers
        self.feature_buffer.append(feat_vector)
        self.timestamp_buffer.append(timestamp_s)
        self.frame_idx_buffer.append(f_idx)

        # 2. Update Heuristic FSM
        fsm_transition_event = self._step_heuristic_fsm(data, timestamp_s, f_idx)

        # 3. Evaluate Frozen GRU (when buffer full and on stride cadence)
        ml_prob = None
        new_provisional_started = False
        if len(self.feature_buffer) == self.temporal_window and (self.session_frame_count % self.stride == 0):
            ml_prob, new_provisional_started = self._step_ml_inference()

        # 4. Hybrid Decision Gate (Strategy H5)
        confirmed_alert = None
        rejected_alert = None

        # Check for confirmation via heuristic transition or recent heuristic falls
        is_static_lying = (
            self.heuristic_detector.current_state in ("LYING_DOWN", "SITTING")
            and self.last_confirmed_fall_time is not None
            and (timestamp_s - self.last_confirmed_fall_time) < self.cooldown_s
        )

        if not is_static_lying:
            for p in list(self.provisional_events):
                ms = p["ml_trigger_time"]
                me = p["ml_end_time"]
                
                matched_h = None
                all_candidates = list(self.recent_heuristic_falls) + list(self.recent_early_transitions)
                for h in all_candidates:
                    hs = h["start"]
                    he = h["time"]
                    if max(hs, ms) <= min(max(he, hs + 0.1), me + self.confirmation_window_s):
                        matched_h = h
                        break

                if matched_h:
                    # CONFIRMED FALL!
                    confirmed_alert = {
                        "event_id": f"FALL_{self.current_session_id}_{len(self.event_log)+1:03d}",
                        "session_id": self.current_session_id,
                        "status": "CONFIRMED_FALL",
                        "ml_trigger_time": round(ms, 3),
                        "ml_end_time": round(me, 3),
                        "heuristic_confirmation_time": round(matched_h["time"], 3),
                        "final_alert_time": round(ms, 3),  # Preserves early warning timestamp!
                        "advance_lead_time_seconds": round(matched_h["time"] - ms, 3),
                        "event_duration_seconds": round(max(me, matched_h["time"]) - ms, 3),
                        "peak_ml_probability": round(max(p["probabilities"]), 4),
                        "mean_ml_probability": round(float(np.mean(p["probabilities"])), 4),
                        "confirmation_delay_seconds": round(timestamp_s - ms, 3),
                        "fsm_state": self.heuristic_detector.current_state,
                    }
                    self.event_log.append(confirmed_alert)
                    self.last_confirmed_fall_time = timestamp_s
                    self.last_confirmed_heuristic_time = matched_h["time"]
                    self.provisional_events.clear()
                    self.current_ml_run = None
                    self.consecutive_positive_windows = 0
                    self.positive_window_history.clear()
                    break

        # Check for provisional event timeout / rejection
        for p in list(self.provisional_events):
            expiration_time = p["ml_end_time"] + self.confirmation_window_s
            if timestamp_s > expiration_time:
                # Discard provisional event as rejected ML false alarm
                rejected_alert = {
                    "event_id": f"REJ_{self.current_session_id}_{len(self.event_log)+1:03d}",
                    "session_id": self.current_session_id,
                    "status": "REJECTED_ML_FALSE_ALARM",
                    "ml_trigger_time": round(p["ml_trigger_time"], 3),
                    "ml_end_time": round(p["ml_end_time"], 3),
                    "expiration_time": round(expiration_time, 3),
                    "peak_ml_probability": round(max(p["probabilities"]), 4),
                    "rejection_reason": f"No heuristic confirmation within W={self.confirmation_window_s:.1f}s confirmation window (expired at {expiration_time:.2f}s, current time {timestamp_s:.2f}s)",
                }
                self.event_log.append(rejected_alert)
                self.provisional_events.remove(p)
                if self.current_ml_run is p:
                    self.current_ml_run = None

        return {
            "session_id": self.current_session_id,
            "frame_number": f_idx,
            "timestamp_seconds": timestamp_s,
            "fsm_state": self.heuristic_detector.current_state,
            "has_person": data["has_person"],
            "raw_kinematic_score": data["raw_score"],
            "ml_probability": ml_prob,
            "provisional_active": self.active_provisional_event is not None,
            "confirmed_fall": confirmed_alert,
            "rejected_alarm": rejected_alert,
            "box": data.get("box"),
            "keypoints": data.get("keypoints"),
            "track_id": data.get("track_id"),
            "all_boxes": data.get("all_boxes"),
        }

    def _step_heuristic_fsm(self, data: Dict[str, Any], ts: float, f_idx: int) -> Optional[Dict[str, Any]]:
        """
        Executes one step of the Refined Finite State Machine.
        Returns a fall confirmation event dict when an active fall transition occurs.
        """
        hd = self.heuristic_detector
        fps = 30.0  # nominal FPS estimate

        # Update upright history memory
        hd.upright_history.append((ts, data["is_upright"]))
        cutoff = ts - hd.upright_lookback
        hd.upright_history = [item for item in hd.upright_history if item[0] >= cutoff]

        was_recently_upright = any(item[1] for item in hd.upright_history)
        had_velocity_spike = data["v_down"] >= hd.vel_thresh
        is_meaningful_descent = data["v_down"] >= self.min_fall_vel

        # Track consecutive downward motion frames
        if is_meaningful_descent:
            self.downward_motion_frames += 1
        else:
            self.downward_motion_frames = max(0, self.downward_motion_frames - 1)

        # Biomechanical Knee Angle Gate & Leg Extension Support:
        # Check knee angles (hip-knee-ankle) ONLY when keypoints have confidence > 0.30
        knee_gate_satisfied = True
        legs_extended = False
        kpts = data.get("keypoints")
        kconfs = data.get("keypoints_conf")
        if kpts is not None and kconfs is not None:
            knee_angles = []
            # Left leg: hip=11, knee=13, ankle=15
            if (len(kpts) > 15 and len(kconfs) > 15
                and kconfs[11] > 0.30 and kconfs[13] > 0.30 and kconfs[15] > 0.30):
                knee_angles.append(self.heuristic_detector._compute_joint_angle(kpts[11], kpts[13], kpts[15]))
            # Right leg: hip=12, knee=14, ankle=16
            if (len(kpts) > 16 and len(kconfs) > 16
                and kconfs[12] > 0.30 and kconfs[14] > 0.30 and kconfs[16] > 0.30):
                knee_angles.append(self.heuristic_detector._compute_joint_angle(kpts[12], kpts[14], kpts[16]))
            
            if len(knee_angles) > 0:
                avg_knee = float(np.mean(knee_angles))
                knee_gate_satisfied = (avg_knee <= self.sitting_knee_angle_max)
                legs_extended = (avg_knee >= self.bending_knee_angle_min)

        # Normal Bending posture check:
        # 1. Torso tilted substantially: angle_deg >= bending_angle_min (35.0 deg)
        # 2. Supported by legs: legs extended (knee >= 135 deg) OR standing aspect ratio on feet
        # 3. Not collapsed on floor: ar >= bending_ar_min (0.85)
        is_bending = (
            (data["angle_deg"] >= self.bending_angle_min)
            and (legs_extended or (data["ar"] >= 1.05 and data["bottom_y"] >= 0.40))
            and (data["ar"] >= self.bending_ar_min)
        )
        effective_horizontal = data["is_horizontal"] and not is_bending

        # Early falling transition check:
        # 1. Was recently upright
        # 2. Meaningful downward velocity (v_down >= min_fall_vel)
        # 3. Early postural transition (torso tilting >= early_angle_thresh OR aspect ratio compressing <= early_ar_thresh)
        # 4. Temporal continuation (at least 2 consecutive downward frames)
        # 5. Gated to exclude normal bending (not is_bending)
        is_early_fall_motion = (
            was_recently_upright
            and is_meaningful_descent
            and (data["angle_deg"] >= self.early_angle_thresh or data["ar"] <= self.early_ar_thresh)
            and self.downward_motion_frames >= 2
            and not is_bending
        )

        if is_early_fall_motion:
            self.recent_early_transitions.append({
                "start": ts - (self.downward_motion_frames / fps),
                "time": ts,
                "type": "EARLY_FALL_TRANSITION",
            })

        # Sitting posture check:
        # 1. AR compressed: sitting_ar_min <= ar <= sitting_ar_max
        # 2. Torso upright: angle_deg <= sitting_angle_max
        # 3. Motion settled: abs(v_down) <= sitting_vel_max
        # 4. Elevated / sitting height: bottom_y >= sitting_min_bottom_y
        # 5. Biomechanical knee gate: knees bent <= sitting_knee_angle_max (if visible)
        is_sitting_posture = (
            (self.sitting_ar_min <= data["ar"] <= self.sitting_ar_max)
            and (data["angle_deg"] <= self.sitting_angle_max)
            and (abs(data["v_down"]) <= self.sitting_vel_max)
            and (data["bottom_y"] >= self.sitting_min_bottom_y)
            and not effective_horizontal
            and knee_gate_satisfied
        )

        if is_sitting_posture:
            if hd.sitting_start_time is None:
                hd.sitting_start_time = ts
            sitting_duration = ts - hd.sitting_start_time
            self.recent_early_transitions.clear()
            self.downward_motion_frames = 0
        else:
            hd.sitting_start_time = None
            sitting_duration = 0.0

        is_confirmed_sitting = (hd.sitting_start_time is not None and sitting_duration >= self.sitting_min_dwell_s)

        # Initial observation / start of video:
        # If horizontal without prior upright posture or velocity spike -> LYING_DOWN
        if not was_recently_upright and hd.current_state == "NORMAL":
            if effective_horizontal and not had_velocity_spike and not is_meaningful_descent:
                hd.current_state = "LYING_DOWN"

        event = None

        if hd.current_state == "NORMAL":
            if is_confirmed_sitting:
                hd.current_state = "SITTING"
                hd.upright_history.clear()
                self.downward_motion_frames = 0
                self.recent_early_transitions.clear()
                self.recent_heuristic_falls.clear()
                self.provisional_events.clear()
                self.current_ml_run = None
                self.consecutive_positive_windows = 0
            else:
                # Condition to initiate FALLING requires dynamic descent evidence:
                # 1. High kinematic score + velocity spike (had_velocity_spike and not is_bending)
                # 2. OR rapid upright-to-horizontal collapse with meaningful descent (was_recently_upright and effective_horizontal and is_meaningful_descent)
                # 3. OR early dynamic fall transition with continuation and active kinematics
                has_dynamic_fall_evidence = (
                    (had_velocity_spike and not is_bending)
                    or (was_recently_upright and effective_horizontal and is_meaningful_descent)
                    or (is_early_fall_motion and (data["raw_score"] >= 0.35 or had_velocity_spike))
                )

                if has_dynamic_fall_evidence and (data["raw_score"] >= 0.35 or had_velocity_spike) and not is_bending:
                    hd.current_state = "FALLING"
                    hd.fall_start_time = ts
                    hd.fall_peak_score = max(data["raw_score"], 0.6)
                    hd.stable_since_time = None
                    event = {
                        "type": "FALL_STARTED",
                        "timestamp_seconds": ts,
                        "frame_number": f_idx,
                        "confidence": max(data["raw_score"], 0.6),
                    }
                elif effective_horizontal and not is_bending:
                    # Intentional / slow lying down without dynamic fall -> LYING_DOWN directly
                    hd.current_state = "LYING_DOWN"

        elif hd.current_state == "SITTING":
            # 1. Person stands back up
            if data["is_upright"] and data["ar"] >= hd.upright_ar:
                hd.current_state = "NORMAL"
                hd.sitting_start_time = None
            # 2. Intentional recline onto bed / floor -> LYING_DOWN directly (no fall alert)
            elif effective_horizontal:
                hd.current_state = "LYING_DOWN"
                hd.sitting_start_time = None
                self.provisional_events.clear()
                self.current_ml_run = None
                self.consecutive_positive_windows = 0
            # 3. Person experiences a genuine fall off bed/chair to the floor
            elif had_velocity_spike and (data["bottom_y"] >= hd.ground_y or data["raw_score"] >= 0.35):
                hd.current_state = "FALLING"
                hd.fall_start_time = ts
                hd.fall_peak_score = max(data["raw_score"], 0.6)
                hd.stable_since_time = None
                event = {
                    "type": "FALL_STARTED",
                    "timestamp_seconds": ts,
                    "frame_number": f_idx,
                    "confidence": max(data["raw_score"], 0.6),
                }

        elif hd.current_state == "FALLING":
            hd.fall_peak_score = max(hd.fall_peak_score, data["raw_score"])
            fall_dur = ts - (hd.fall_start_time or ts)

            # Check stabilization on floor
            if effective_horizontal and abs(data["v_down"]) <= hd.stab_vel:
                if hd.stable_since_time is None:
                    hd.stable_since_time = ts
                elif (ts - hd.stable_since_time) >= hd.stab_dur:
                    hd.current_state = "LYING_DOWN"
                    event = {
                        "type": "FALL_STABILIZED_LYING",
                        "timestamp_seconds": hd.fall_start_time if hd.fall_start_time is not None else ts,
                        "end_timestamp_seconds": ts,
                        "frame_number": f_idx,
                        "confidence": hd.fall_peak_score,
                    }
            else:
                hd.stable_since_time = None

            # Max duration safety limit
            if hd.current_state == "FALLING" and fall_dur >= hd.max_dur:
                hd.current_state = "LYING_DOWN"
                event = {
                    "type": "FALL_TIMEOUT_LYING",
                    "timestamp_seconds": hd.fall_start_time if hd.fall_start_time is not None else ts,
                    "end_timestamp_seconds": ts,
                    "frame_number": f_idx,
                    "confidence": hd.fall_peak_score,
                }

            # Recovery
            if data["is_upright"]:
                hd.current_state = "NORMAL"
                hd.stable_since_time = None
                hd.fall_start_time = None
                hd.sitting_start_time = None

        elif hd.current_state == "LYING_DOWN":
            if data["is_upright"]:
                hd.current_state = "NORMAL"
                hd.stable_since_time = None
                hd.fall_start_time = None
                hd.sitting_start_time = None

        if event:
            hd_start = event.get("timestamp_seconds", ts)
            self.recent_heuristic_falls.append({
                "start": hd_start,
                "time": ts,
                "type": event["type"],
            })

        return event

    def _step_ml_inference(self) -> Tuple[float, bool]:
        """
        Runs the frozen FallGRUClassifier on the rolling 32-frame buffer.
        Manages consecutive runs (M=2) and provisional events.
        """
        # Shape: [1, 32, 62]
        window_feats = np.stack(list(self.feature_buffer), axis=0)[np.newaxis, :, :]
        tensor_x = torch.from_numpy(window_feats).float().to(self.device)

        with torch.no_grad():
            logits = self.gru_model(tensor_x)
            prob = float(torch.sigmoid(logits).item())

        win_start_t = self.timestamp_buffer[0]
        win_end_t = self.timestamp_buffer[-1]
        win_start_f = self.frame_idx_buffer[0]
        win_end_f = self.frame_idx_buffer[-1]

        new_provisional_started = False

        # Suppress spawning new provisional triggers during static lying or post-fall cooldown:
        # 1. While statically lying down (whether pre-existing, intentional, or post-fall)
        # 2. During post-fall cooldown window
        is_static_lying_suppression = (
            self.suppress_during_static_lying
            and self.heuristic_detector.current_state in ("LYING_DOWN", "SITTING")
        )
        is_post_fall_cooldown = (
            self.last_confirmed_fall_time is not None
            and (win_end_t - self.last_confirmed_fall_time) < self.cooldown_s
        )

        if is_static_lying_suppression or is_post_fall_cooldown:
            self.consecutive_positive_windows = 0
            self.current_ml_run = None
            return prob, False

        if prob >= self.ml_threshold:
            self.consecutive_positive_windows += 1
            self.positive_window_history.append({
                "start_time": win_start_t,
                "end_time": win_end_t,
                "start_frame": win_start_f,
                "end_frame": win_end_f,
                "probability": prob,
            })

            # Check if M consecutive threshold reached
            if self.consecutive_positive_windows >= self.postprocess_M:
                if self.current_ml_run is None:
                    # Look back to the start of this positive run (M windows back)
                    earliest_win = self.positive_window_history[-self.consecutive_positive_windows]
                    new_run = {
                        "session_id": self.current_session_id,
                        "ml_trigger_time": earliest_win["start_time"],
                        "ml_end_time": win_end_t,
                        "start_frame": earliest_win["start_frame"],
                        "end_frame": win_end_f,
                        "probabilities": [w["probability"] for w in self.positive_window_history[-self.consecutive_positive_windows:]],
                        "status": "PROVISIONAL",
                    }
                    self.current_ml_run = new_run
                    self.provisional_events.append(new_run)
                    new_provisional_started = True
                else:
                    # Extend end time of ongoing ML activation
                    self.current_ml_run["ml_end_time"] = win_end_t
                    self.current_ml_run["end_frame"] = win_end_f
                    self.current_ml_run["probabilities"].append(prob)
        else:
            self.consecutive_positive_windows = 0
            self.current_ml_run = None

        return prob, new_provisional_started

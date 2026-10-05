"""
fall_detector_refined.py - Refined Fall Detector with Dynamic State Machine.

Implements:
  - 3-State Finite State Machine: NORMAL -> FALLING -> LYING_DOWN
  - Explicit dynamic transition requirement (descent velocity spike OR rapid upright-to-horizontal transition)
  - True stabilization detection to bound the dynamic fall event to ~1-3 seconds
  - Floor-start rejection (initial horizontal posture at t=0 defaults to LYING_DOWN, not FALLING)
  - Preservation of repeated falls (NORMAL -> FALLING -> LYING_DOWN -> NORMAL -> FALLING...)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

COCO_SKELETON = [
    (5, 6),   # Shoulders
    (5, 7), (7, 9),   # Left Arm
    (6, 8), (8, 10),  # Right Arm
    (5, 11), (6, 12), # Torso sides
    (11, 12),         # Hips
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16), # Right Leg
]


class RefinedFallDetector:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        weights = self.cfg["model"]["weights"]
        self.model = YOLO(weights)
        self.conf_thresh = self.cfg["model"]["conf_threshold"]
        self.iou_thresh = self.cfg["model"]["iou_threshold"]
        self.img_size = self.cfg["model"]["img_size"]

        k = self.cfg["kinematics"]
        self.upright_ar = k["upright_aspect_ratio"]
        self.upright_angle = k["upright_torso_angle"]
        self.horiz_ar = k["horizontal_aspect_ratio"]
        self.horiz_angle = k["horizontal_torso_angle"]
        self.vel_thresh = k["dynamic_velocity_thresh"]
        self.ground_y = k["ground_plane_y_thresh"]
        self.min_fall_vel = k.get("min_fall_velocity", 0.15)
        self.early_angle = k.get("early_angle_threshold", 22.0)
        self.early_ar = k.get("early_ar_threshold", 1.25)
        self.sitting_ar_min = k.get("sitting_aspect_ratio_min", 0.70)
        self.sitting_ar_max = k.get("sitting_aspect_ratio_max", 1.05)
        self.sitting_angle_max = k.get("sitting_torso_angle_max", 35.0)
        self.sitting_knee_angle_max = k.get("sitting_knee_angle_max", 130.0)
        self.sitting_vel_max = k.get("sitting_velocity_max", 0.12)
        self.sitting_min_dwell_s = k.get("sitting_dwell_seconds", 0.50)
        self.sitting_min_bottom_y = k.get("sitting_min_bottom_y", 0.40)
        self.bending_angle_min = k.get("bending_torso_angle_min", 35.0)
        self.bending_knee_angle_min = k.get("bending_knee_angle_min", 135.0)
        self.bending_ar_min = k.get("bending_aspect_ratio_min", 0.85)

        tracking_cfg = self.cfg.get("tracking", {})
        self.tracking_enabled = tracking_cfg.get("enabled", True)
        self.max_missed_frames = tracking_cfg.get("max_missed_frames", 15)
        self.iou_reconnect_thresh = tracking_cfg.get("iou_reconnect_thresh", 0.15)
        self.max_center_dist_norm = tracking_cfg.get("max_center_dist_norm", 0.25)

        w = self.cfg.get("weights", {})
        self.w_angle = w.get("w_angle", 0.40)
        self.w_ar = w.get("w_aspect_ratio", 0.30)
        self.w_floor = w.get("w_floor", 0.15)
        self.w_vel = w.get("w_velocity", 0.15)

        sm = self.cfg["state_machine"]
        self.upright_lookback = sm["upright_lookback_seconds"]
        self.stab_vel = sm["stabilization_velocity_thresh"]
        self.stab_dur = sm["stabilization_duration_seconds"]
        self.min_dur = sm["min_fall_event_duration_seconds"]
        self.max_dur = sm["max_fall_event_duration_seconds"]
        self.merge_gap = sm["merge_gap_seconds"]
        self.frame_thresh = sm["frame_fall_threshold"]
        self.video_conf_thresh = sm["video_fall_min_confidence"]

        self.reset()

    def reset(self):
        self.cog_history: List[Tuple[float, float]] = []  # (ts, cog_y)
        self.upright_history: List[Tuple[float, bool]] = []  # (ts, is_upright)
        self.downward_motion_frames: int = 0
        self.current_state = "NORMAL"  # NORMAL, SITTING, FALLING, LYING_DOWN
        self.sitting_start_time: Optional[float] = None
        self.fall_start_time: Optional[float] = None
        self.fall_peak_score: float = 0.0
        self.stable_since_time: Optional[float] = None
        self.last_valid_result: Optional[Dict] = None
        self.target_track_id: Optional[int] = None
        self.target_box: Optional[np.ndarray] = None
        self.target_missing_frames: int = 0
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers is not None:
                for t in self.model.predictor.trackers:
                    if hasattr(t, "reset"):
                        t.reset()

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH
        areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        unionArea = areaA + areaB - interArea
        return float(interArea / unionArea) if unionArea > 0 else 0.0

    @staticmethod
    def _compute_center_dist_norm(boxA, boxB, w: int, h: int) -> float:
        cA_x = (boxA[0] + boxA[2]) / (2.0 * max(1, w))
        cA_y = (boxA[1] + boxA[3]) / (2.0 * max(1, h))
        cB_x = (boxB[0] + boxB[2]) / (2.0 * max(1, w))
        cB_y = (boxB[1] + boxB[3]) / (2.0 * max(1, h))
        return float(np.hypot(cA_x - cB_x, cA_y - cB_y))

    def _associate_target_person(
        self, res, h: int, w: int
    ) -> Tuple[Optional[int], Optional[int], List[Dict[str, Any]]]:
        """
        Associates detections with the locked target person.
        Returns:
            (selected_idx, target_track_id, all_boxes)
        """
        n_det = len(res.boxes)
        if n_det == 0:
            self.target_missing_frames += 1
            if self.target_missing_frames > self.max_missed_frames:
                self.target_track_id = None
                self.target_box = None
                self.target_missing_frames = 0
                self.current_state = "NORMAL"
                self.cog_history = []
            return None, self.target_track_id, []

        detected_persons = []
        for i in range(n_det):
            box = res.boxes[i].xyxy[0].cpu().numpy()
            conf = float(res.boxes[i].conf[0].cpu().numpy()) if hasattr(res.boxes[i], "conf") and res.boxes[i].conf is not None else 0.0
            tid = None
            if hasattr(res.boxes, "id") and res.boxes.id is not None and len(res.boxes.id) > i and res.boxes.id[i] is not None:
                try:
                    tid = int(res.boxes.id[i].item())
                except Exception:
                    tid = None

            kpts = None
            kconf = None
            if hasattr(res, "keypoints") and res.keypoints is not None:
                if hasattr(res.keypoints, "xy") and res.keypoints.xy is not None and len(res.keypoints.xy) > i:
                    try:
                        kpts = res.keypoints.xy[i].cpu().numpy()
                    except Exception:
                        kpts = None
                if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None and len(res.keypoints.conf) > i:
                    try:
                        kconf = res.keypoints.conf[i].cpu().numpy()
                    except Exception:
                        kconf = None

            detected_persons.append({
                "index": i,
                "box": box,
                "conf": conf,
                "track_id": tid,
                "keypoints": kpts,
                "keypoints_conf": kconf,
            })

        if not self.tracking_enabled:
            best_idx = int(np.argmax([p["conf"] for p in detected_persons]))
            all_boxes = [
                {
                    "box": p["box"],
                    "track_id": None,
                    "conf": p["conf"],
                    "is_target": (p["index"] == best_idx),
                    "keypoints": p["keypoints"],
                    "keypoints_conf": p["keypoints_conf"],
                }
                for p in detected_persons
            ]
            return best_idx, None, all_boxes

        # 1. Uninitialized target: lock onto most confident person
        if self.target_track_id is None and self.target_box is None:
            best_idx = int(np.argmax([p["conf"] for p in detected_persons]))
            chosen = detected_persons[best_idx]
            self.target_track_id = chosen["track_id"]
            self.target_box = chosen["box"]
            self.target_missing_frames = 0
            all_boxes = [
                {
                    "box": p["box"],
                    "track_id": p["track_id"],
                    "conf": p["conf"],
                    "is_target": (p["index"] == best_idx),
                    "keypoints": p["keypoints"],
                    "keypoints_conf": p["keypoints_conf"],
                }
                for p in detected_persons
            ]
            return best_idx, self.target_track_id, all_boxes

        # 2. Primary Track-ID Association
        matched_idx = None
        if self.target_track_id is not None:
            for p in detected_persons:
                if p["track_id"] == self.target_track_id:
                    matched_idx = p["index"]
                    break

        # 3. Spatial IoU / Center Distance Fallback if track ID dropped or reassigned
        if matched_idx is None and self.target_box is not None:
            best_iou = -1.0
            best_candidate = None
            for p in detected_persons:
                iou = self._compute_iou(self.target_box, p["box"])
                dist = self._compute_center_dist_norm(self.target_box, p["box"], w, h)
                if iou >= self.iou_reconnect_thresh or dist <= self.max_center_dist_norm:
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = p

            if best_candidate is not None:
                matched_idx = best_candidate["index"]
                if best_candidate["track_id"] is not None:
                    self.target_track_id = best_candidate["track_id"]

        # 4. Handle association outcome
        if matched_idx is not None:
            chosen = detected_persons[matched_idx]
            self.target_box = chosen["box"]
            self.target_missing_frames = 0
            selected_idx = matched_idx
        else:
            self.target_missing_frames += 1
            if self.target_missing_frames > self.max_missed_frames:
                best_idx = int(np.argmax([p["conf"] for p in detected_persons]))
                chosen = detected_persons[best_idx]
                self.target_track_id = chosen["track_id"]
                self.target_box = chosen["box"]
                self.target_missing_frames = 0
                self.current_state = "NORMAL"
                self.cog_history = []
                selected_idx = best_idx
            else:
                selected_idx = None

        all_boxes = [
            {
                "box": p["box"],
                "track_id": p["track_id"],
                "conf": p["conf"],
                "is_target": (p["index"] == selected_idx),
                "keypoints": p["keypoints"],
                "keypoints_conf": p["keypoints_conf"],
            }
            for p in detected_persons
        ]
        return selected_idx, self.target_track_id, all_boxes

    @staticmethod
    def _compute_joint_angle(a, b, c) -> float:
        """Computes joint angle ABC (in degrees) at joint vertex b."""
        ba = np.array(a[:2], dtype=float) - np.array(b[:2], dtype=float)
        bc = np.array(c[:2], dtype=float) - np.array(b[:2], dtype=float)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-5 or norm_bc < 1e-5:
            return 180.0
        cos_ang = np.dot(ba, bc) / (norm_ba * norm_bc)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    def analyze_frame_kinematics(self, frame: np.ndarray, timestamp_s: float) -> Dict:
        """
        Extract raw posture and velocity features from frame with persistent target tracking.
        """
        h, w = frame.shape[:2]
        if self.tracking_enabled and hasattr(self.model, "track"):
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.img_size,
                    device="cpu",
                )
            except Exception:
                results = self.model.predict(
                    frame,
                    verbose=False,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.img_size,
                    device="cpu",
                )
        else:
            results = self.model.predict(
                frame,
                verbose=False,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                device="cpu",
            )
        res = results[0]

        data = {
            "timestamp_s": timestamp_s,
            "has_person": False,
            "track_id": self.target_track_id,
            "all_boxes": [],
            "box": None,
            "keypoints": None,
            "keypoints_conf": None,
            "ar": 1.5,
            "angle_deg": 10.0,
            "bottom_y": 0.5,
            "v_down": 0.0,
            "is_upright": True,
            "is_horizontal": False,
            "raw_score": 0.0,
        }

        selected_idx, track_id, all_boxes = self._associate_target_person(res, h, w)
        data["track_id"] = track_id
        data["all_boxes"] = all_boxes

        if selected_idx is None:
            if self.last_valid_result:
                # Carry over target's last posture with zero velocity
                data["ar"] = self.last_valid_result["ar"]
                data["angle_deg"] = self.last_valid_result["angle_deg"]
                data["bottom_y"] = self.last_valid_result["bottom_y"]
                data["is_upright"] = self.last_valid_result["is_upright"]
                data["is_horizontal"] = self.last_valid_result["is_horizontal"]
                data["raw_score"] = self.last_valid_result["raw_score"] * 0.95
                data["box"] = self.target_box
                data["keypoints"] = self.last_valid_result.get("keypoints")
                data["keypoints_conf"] = self.last_valid_result.get("keypoints_conf")
            return data

        data["has_person"] = True
        box = res.boxes[selected_idx].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        ar = bh / bw
        cog_y = (y1 + y2) / (2.0 * h)
        bottom_y = y2 / float(h)

        data["box"] = box
        data["ar"] = ar
        data["bottom_y"] = bottom_y

        # Velocity tracking over window (strictly for the locked target)
        self.cog_history.append((timestamp_s, cog_y))
        if len(self.cog_history) > 12:
            self.cog_history.pop(0)

        v_down = 0.0
        if len(self.cog_history) >= 4:
            dt = max(0.01, self.cog_history[-1][0] - self.cog_history[0][0])
            dy = self.cog_history[-1][1] - self.cog_history[0][1]
            v_down = dy / dt
        data["v_down"] = v_down

        # Keypoints & Torso Orientation (strictly for the locked target)
        has_kpts = False
        angle_deg = 0.0
        if res.keypoints is not None and len(res.keypoints.xy) > selected_idx:
            kpts = res.keypoints.xy[selected_idx].cpu().numpy()
            confs = (
                res.keypoints.conf[selected_idx].cpu().numpy()
                if res.keypoints.conf is not None
                else np.ones(17)
            )
            data["keypoints"] = kpts
            data["keypoints_conf"] = confs

            sh_pts = [kpts[i] for i in (5, 6) if confs[i] > 0.25 and kpts[i].sum() > 0]
            hip_pts = [kpts[i] for i in (11, 12) if confs[i] > 0.25 and kpts[i].sum() > 0]

            if sh_pts and hip_pts:
                has_kpts = True
                mid_sh = np.mean(sh_pts, axis=0)
                mid_hip = np.mean(hip_pts, axis=0)
                dx = abs(float(mid_sh[0] - mid_hip[0]))
                dy = abs(float(mid_sh[1] - mid_hip[1]))
                angle_deg = float(np.degrees(np.arctan2(dx, dy + 1e-5)))

        data["angle_deg"] = angle_deg

        # Check legs extended / supported by legs if keypoints available
        legs_extended = False
        kpts_list = data.get("keypoints")
        confs_list = data.get("keypoints_conf")
        if kpts_list is not None and confs_list is not None:
            knee_angles = []
            if (len(kpts_list) > 15 and len(confs_list) > 15
                and confs_list[11] > 0.30 and confs_list[13] > 0.30 and confs_list[15] > 0.30):
                knee_angles.append(self._compute_joint_angle(kpts_list[11], kpts_list[13], kpts_list[15]))
            if (len(kpts_list) > 16 and len(confs_list) > 16
                and confs_list[12] > 0.30 and confs_list[14] > 0.30 and confs_list[16] > 0.30):
                knee_angles.append(self._compute_joint_angle(kpts_list[12], kpts_list[14], kpts_list[16]))
            if len(knee_angles) > 0:
                avg_knee = float(np.mean(knee_angles))
                legs_extended = (avg_knee >= self.bending_knee_angle_min)

        is_bending = (
            (angle_deg >= self.bending_angle_min)
            and (legs_extended or (ar >= 1.05 and bottom_y >= 0.40))
            and (ar >= self.bending_ar_min)
        )
        data["is_bending"] = is_bending

        # Posture boolean states
        is_upright = (ar >= self.upright_ar) and (not has_kpts or angle_deg <= self.upright_angle)
        is_horizontal = (
            (ar <= self.horiz_ar or (has_kpts and angle_deg >= self.horiz_angle))
            and (bottom_y >= 0.35)
            and not is_bending
        )
        data["is_upright"] = is_upright
        data["is_horizontal"] = is_horizontal

        # Raw frame kinematics score
        s_ar = float(np.clip((self.upright_ar - ar) / (self.upright_ar - self.horiz_ar), 0.0, 1.0))
        s_angle = (
            float(np.clip((angle_deg - self.upright_angle) / (self.horiz_angle - self.upright_angle), 0.0, 1.0))
            if has_kpts
            else s_ar
        )
        s_floor = float(np.clip((bottom_y - 0.50) / (self.ground_y - 0.50), 0.0, 1.0))
        s_vel = float(np.clip(v_down / self.vel_thresh, 0.0, 1.0))

        if has_kpts:
            raw_score = (
                self.w_angle * s_angle
                + self.w_ar * s_ar
                + self.w_floor * s_floor
                + self.w_vel * s_vel
            )
        else:
            raw_score = (
                (self.w_ar + self.w_angle * 0.5) * s_ar
                + (self.w_floor + self.w_angle * 0.25) * s_floor
                + (self.w_vel + self.w_angle * 0.25) * s_vel
            )
        data["raw_score"] = float(np.clip(raw_score, 0.0, 1.0))

        self.last_valid_result = data
        return data

    def process_video_sequence(
        self, frames_generator, timestamps_s: List[float], fps: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run the full temporal finite state machine over all frames of a video.

        Returns:
            (frame_records, fall_events)
        """
        self.reset()
        total_frames = len(timestamps_s)

        frame_records: List[Dict] = []
        raw_events: List[Dict] = []

        active_event_start_idx: Optional[int] = None
        active_event_start_t: Optional[float] = None
        active_event_peak_score: float = 0.0

        for f_idx, frame in enumerate(frames_generator):
            ts = timestamps_s[f_idx] if f_idx < total_frames else f_idx / fps
            data = self.analyze_frame_kinematics(frame, ts)

            # Update upright memory buffer
            self.upright_history.append((ts, data["is_upright"]))
            # Keep only entries within lookback window
            cutoff = ts - self.upright_lookback
            self.upright_history = [item for item in self.upright_history if item[0] >= cutoff]

            was_recently_upright = any(item[1] for item in self.upright_history)
            had_velocity_spike = data["v_down"] >= self.vel_thresh
            is_meaningful_descent = data["v_down"] >= self.min_fall_vel

            if is_meaningful_descent:
                self.downward_motion_frames += 1
            else:
                self.downward_motion_frames = max(0, self.downward_motion_frames - 1)

            # Biomechanical Knee Angle Gate & Leg Extension Support:
            # Check knee flexion (hip-knee-ankle) ONLY when keypoints have confidence > 0.30
            knee_gate_satisfied = True
            legs_extended = False
            kpts = data.get("keypoints")
            kconfs = data.get("keypoints_conf")
            if kpts is not None and kconfs is not None:
                knee_angles = []
                # Left leg: hip=11, knee=13, ankle=15
                if (len(kpts) > 15 and len(kconfs) > 15
                    and kconfs[11] > 0.30 and kconfs[13] > 0.30 and kconfs[15] > 0.30):
                    knee_angles.append(self._compute_joint_angle(kpts[11], kpts[13], kpts[15]))
                # Right leg: hip=12, knee=14, ankle=16
                if (len(kpts) > 16 and len(kconfs) > 16
                    and kconfs[12] > 0.30 and kconfs[14] > 0.30 and kconfs[16] > 0.30):
                    knee_angles.append(self._compute_joint_angle(kpts[12], kpts[14], kpts[16]))
                
                if len(knee_angles) > 0:
                    avg_knee = float(np.mean(knee_angles))
                    knee_gate_satisfied = (avg_knee <= self.sitting_knee_angle_max)
                    legs_extended = (avg_knee >= self.bending_knee_angle_min)

            # Normal Bending posture check:
            is_bending = data.get("is_bending", False) or (
                (data["angle_deg"] >= self.bending_angle_min)
                and (legs_extended or (data["ar"] >= 1.05 and data["bottom_y"] >= 0.40))
                and (data["ar"] >= self.bending_ar_min)
            )
            effective_horizontal = data["is_horizontal"] and not is_bending

            # Early falling transition check:
            # 1. Was recently upright
            # 2. Meaningful downward velocity (v_down >= min_fall_vel)
            # 3. Early postural transition (torso tilting >= early_angle OR aspect ratio compressing <= early_ar)
            # 4. Temporal continuation (at least 2 consecutive downward frames)
            # 5. Gated to exclude normal bending (not is_bending)
            is_early_fall_motion = (
                was_recently_upright
                and is_meaningful_descent
                and (data["angle_deg"] >= self.early_angle or data["ar"] <= self.early_ar)
                and self.downward_motion_frames >= 2
                and not is_bending
            )

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
                if self.sitting_start_time is None:
                    self.sitting_start_time = ts
                sitting_duration = ts - self.sitting_start_time
            else:
                self.sitting_start_time = None
                sitting_duration = 0.0

            is_confirmed_sitting = (self.sitting_start_time is not None and sitting_duration >= self.sitting_min_dwell_s)

            # State Machine Transitions
            # Handle start of video / initial observation:
            # If horizontal without prior upright posture or velocity spike -> LYING_DOWN
            if not was_recently_upright and self.current_state == "NORMAL":
                if effective_horizontal and not had_velocity_spike and not is_meaningful_descent:
                    self.current_state = "LYING_DOWN"

            if self.current_state == "NORMAL":
                if is_confirmed_sitting:
                    self.current_state = "SITTING"
                    self.upright_history.clear()
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
                        self.current_state = "FALLING"
                        self.fall_start_time = ts
                        self.fall_peak_score = max(data["raw_score"], 0.6)
                        self.stable_since_time = None
                        active_event_start_idx = f_idx
                        active_event_start_t = ts
                    elif effective_horizontal and not is_bending:
                        # Intentional / slow lying down without dynamic fall -> LYING_DOWN directly
                        self.current_state = "LYING_DOWN"

            elif self.current_state == "SITTING":
                # 1. Person stands back up
                if data["is_upright"] and data["ar"] >= self.upright_ar:
                    self.current_state = "NORMAL"
                    self.sitting_start_time = None
                # 2. Intentional recline onto bed / floor -> LYING_DOWN directly (no fall alert)
                elif effective_horizontal:
                    self.current_state = "LYING_DOWN"
                    self.sitting_start_time = None
                # 3. Person experiences a genuine fall off bed/chair to the floor
                elif had_velocity_spike and (data["bottom_y"] >= self.ground_y or data["raw_score"] >= 0.35):
                    self.current_state = "FALLING"
                    self.fall_start_time = ts
                    self.fall_peak_score = max(data["raw_score"], 0.6)
                    self.stable_since_time = None
                    active_event_start_idx = f_idx
                    active_event_start_t = ts

            elif self.current_state == "FALLING":
                active_event_peak_score = max(active_event_peak_score, data["raw_score"])
                fall_duration = ts - (self.fall_start_time or ts)

                # Check for stabilization on floor
                # Downward velocity settled to near zero AND horizontal
                if effective_horizontal and abs(data["v_down"]) <= self.stab_vel:
                    if self.stable_since_time is None:
                        self.stable_since_time = ts
                    elif (ts - self.stable_since_time) >= self.stab_dur:
                        # Fall transition finished! Transition to LYING_DOWN
                        self.current_state = "LYING_DOWN"
                        if active_event_start_t is not None:
                            event_dur = ts - active_event_start_t
                            if event_dur >= self.min_dur:
                                raw_events.append({
                                    "start_idx": active_event_start_idx,
                                    "end_idx": f_idx,
                                    "start_time_seconds": active_event_start_t,
                                    "end_time_seconds": ts,
                                    "confidence": active_event_peak_score,
                                })
                        active_event_start_idx = None
                        active_event_start_t = None
                else:
                    self.stable_since_time = None

                # Safety limit check: max_fall_event_duration_seconds
                if self.current_state == "FALLING" and fall_duration >= self.max_dur:
                    self.current_state = "LYING_DOWN"
                    if active_event_start_t is not None:
                        event_dur = ts - active_event_start_t
                        if event_dur >= self.min_dur:
                            raw_events.append({
                                "start_idx": active_event_start_idx,
                                "end_idx": f_idx,
                                "start_time_seconds": active_event_start_t,
                                "end_time_seconds": ts,
                                "confidence": active_event_peak_score,
                            })
                    active_event_start_idx = None
                    active_event_start_t = None

                # If person gets back up while falling (recovery)
                if data["is_upright"]:
                    if active_event_start_t is not None:
                        event_dur = ts - active_event_start_t
                        if event_dur >= self.min_dur:
                            raw_events.append({
                                "start_idx": active_event_start_idx,
                                "end_idx": f_idx,
                                "start_time_seconds": active_event_start_t,
                                "end_time_seconds": ts,
                                "confidence": active_event_peak_score,
                            })
                    self.current_state = "NORMAL"
                    active_event_start_idx = None
                    active_event_start_t = None
                    self.stable_since_time = None
                    self.fall_start_time = None
                    self.sitting_start_time = None

            elif self.current_state == "LYING_DOWN":
                # Check if person stands back up
                if data["is_upright"]:
                    self.current_state = "NORMAL"
                    self.stable_since_time = None
                    self.fall_start_time = None
                    self.sitting_start_time = None

            # Frame label and confidence
            # A frame is labeled FALL only if currently in active FALLING transition
            is_fall_active = (self.current_state == "FALLING")
            pred_label = "FALL" if is_fall_active else "NO_FALL"
            pred_conf = data["raw_score"] if is_fall_active else round(1.0 - data["raw_score"], 3)

            record = {
                "frame_number": f_idx + 1,
                "timestamp_seconds": ts,
                "state": self.current_state,
                "prediction": pred_label,
                "confidence": pred_conf,
                "raw_score": data["raw_score"],
                "box": data["box"],
                "keypoints": data["keypoints"],
                "keypoints_conf": data["keypoints_conf"],
                "ar": data["ar"],
                "angle_deg": data["angle_deg"],
                "v_down": data["v_down"],
                "bottom_y": data["bottom_y"],
            }
            frame_records.append(record)

        # Handle event if video ended while still in FALLING state
        if active_event_start_t is not None and active_event_start_idx is not None:
            end_t = timestamps_s[-1] if timestamps_s else 0.0
            event_dur = end_t - active_event_start_t
            if event_dur >= self.min_dur:
                raw_events.append({
                    "start_idx": active_event_start_idx,
                    "end_idx": total_frames - 1,
                    "start_time_seconds": active_event_start_t,
                    "end_time_seconds": end_t,
                    "confidence": active_event_peak_score,
                })

        # Merge nearby micro-events separated by <= merge_gap
        merged_events: List[Dict] = []
        if raw_events:
            curr = raw_events[0]
            for nxt in raw_events[1:]:
                gap = nxt["start_time_seconds"] - curr["end_time_seconds"]
                total_dur = nxt["end_time_seconds"] - curr["start_time_seconds"]
                if gap <= self.merge_gap and total_dur <= self.max_dur:
                    curr["end_idx"] = nxt["end_idx"]
                    curr["end_time_seconds"] = nxt["end_time_seconds"]
                    curr["confidence"] = max(curr["confidence"], nxt["confidence"])
                else:
                    merged_events.append(curr)
                    curr = nxt
            merged_events.append(curr)

        final_events = []
        for idx, ev in enumerate(merged_events, start=1):
            final_events.append({
                "event_id": idx,
                "start_time_seconds": round(ev["start_time_seconds"], 2),
                "end_time_seconds": round(ev["end_time_seconds"], 2),
                "confidence": round(ev["confidence"], 3),
                "start_idx": ev["start_idx"],
                "end_idx": ev["end_idx"],
            })

        # Synchronize frame predictions with segmented event bounds
        # (ensures frames within validated events match event boundaries)
        event_frame_map = {}
        for ev in final_events:
            for fi in range(ev["start_idx"], ev["end_idx"] + 1):
                event_frame_map[fi] = ev["event_id"]

        for idx, rec in enumerate(frame_records):
            if idx in event_frame_map:
                rec["prediction"] = "FALL"
                rec["event_id"] = event_frame_map[idx]
                rec["confidence"] = max(rec["confidence"], 0.60)
            else:
                if rec["prediction"] == "FALL":
                    rec["prediction"] = "NO_FALL"
                    rec["confidence"] = round(1.0 - rec["raw_score"], 3)
                rec["event_id"] = 0

        return frame_records, final_events

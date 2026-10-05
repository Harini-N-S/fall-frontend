"""
fall_detector.py - YOLO-Pose & Biomechanical Kinematics Fall Detector.

Implements temporal posture analysis, keypoint torso geometry, aspect ratio tracking,
velocity spikes, and temporal event aggregation.
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


class FallDetector:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        weights = self.cfg["model"]["weights"]
        self.model = YOLO(weights)
        self.conf_thresh = self.cfg["model"]["conf_threshold"]
        self.iou_thresh = self.cfg["model"]["iou_threshold"]
        self.img_size = self.cfg["model"]["img_size"]

        # Kinematic parameters
        k = self.cfg["kinematics"]
        self.ar_fall = k["aspect_ratio_fall_thresh"]
        self.ar_upright = k["aspect_ratio_upright_thresh"]
        self.angle_fall = k["torso_angle_fall_thresh"]
        self.angle_upright = k["torso_angle_upright_thresh"]
        self.vel_thresh = k["velocity_spike_thresh"]
        self.ground_y = k["ground_plane_y_thresh"]

        # Weights
        w = self.cfg.get("weights", {})
        self.w_angle = w.get("w_angle", 0.40)
        self.w_ar = w.get("w_aspect_ratio", 0.30)
        self.w_floor = w.get("w_floor", 0.15)
        self.w_vel = w.get("w_velocity", 0.15)

        # Temporal parameters
        t = self.cfg["temporal"]
        self.frame_thresh = t["frame_fall_threshold"]
        self.min_consecutive = t["min_consecutive_frames"]
        self.min_duration = t["min_event_duration_seconds"]
        self.merge_gap = t["merge_gap_seconds"]
        self.video_conf_thresh = t["video_fall_min_confidence"]

        # History state for velocity
        self.reset_history()

    def reset_history(self):
        self.cog_history: List[Tuple[float, float]] = []  # (timestamp_s, cog_y_norm)
        self.last_valid_result: Optional[Dict] = None

    def process_frame(self, frame: np.ndarray, timestamp_s: float) -> Dict:
        """
        Analyze a single video frame and extract kinematic fall features.
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device="cpu",
        )
        res = results[0]

        frame_data = {
            "timestamp_s": timestamp_s,
            "has_person": False,
            "score": 0.0,
            "box": None,
            "keypoints": None,
            "keypoints_conf": None,
            "ar": 0.0,
            "angle_deg": 0.0,
            "v_down": 0.0,
            "bottom_y": 0.0,
            "details": "No Person",
        }

        if len(res.boxes) == 0:
            # Check if person was previously on floor and occluded
            if self.last_valid_result and self.last_valid_result["score"] > 0.6:
                # Decay score smoothly if person is lying still or partially occluded
                frame_data["score"] = max(0.0, self.last_valid_result["score"] - 0.05)
                frame_data["details"] = "Persisted post-fall state"
            return frame_data

        frame_data["has_person"] = True
        best_idx = 0  # Dominant person in scene
        box = res.boxes[best_idx].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        ar = bh / bw
        cog_y = (y1 + y2) / (2.0 * h)
        bottom_y = y2 / float(h)

        frame_data["box"] = box
        frame_data["ar"] = ar
        frame_data["bottom_y"] = bottom_y

        # Update CoG history and estimate downward velocity
        self.cog_history.append((timestamp_s, cog_y))
        if len(self.cog_history) > 12:
            self.cog_history.pop(0)

        v_down = 0.0
        if len(self.cog_history) >= 4:
            dt = max(0.01, self.cog_history[-1][0] - self.cog_history[0][0])
            dy = self.cog_history[-1][1] - self.cog_history[0][1]
            v_down = dy / dt  # Positive is downward in screen coordinates
        frame_data["v_down"] = v_down

        # Extract Keypoints & Torso Orientation
        has_kpts = False
        angle_deg = 0.0
        if res.keypoints is not None and len(res.keypoints.xy) > best_idx:
            kpts = res.keypoints.xy[best_idx].cpu().numpy()  # (17, 2)
            confs = (
                res.keypoints.conf[best_idx].cpu().numpy()
                if res.keypoints.conf is not None
                else np.ones(17)
            )
            frame_data["keypoints"] = kpts
            frame_data["keypoints_conf"] = confs

            sh_pts = [kpts[i] for i in (5, 6) if confs[i] > 0.25 and kpts[i].sum() > 0]
            hip_pts = [kpts[i] for i in (11, 12) if confs[i] > 0.25 and kpts[i].sum() > 0]

            if sh_pts and hip_pts:
                has_kpts = True
                mid_sh = np.mean(sh_pts, axis=0)
                mid_hip = np.mean(hip_pts, axis=0)
                dx = abs(float(mid_sh[0] - mid_hip[0]))
                dy = abs(float(mid_sh[1] - mid_hip[1]))
                angle_deg = float(np.degrees(np.arctan2(dx, dy + 1e-5)))

        frame_data["angle_deg"] = angle_deg

        # Calculate Normalized Metric Indicators [0.0, 1.0]
        # 1. Aspect Ratio: 1.0 when lying (ar <= ar_fall), 0.0 when standing (ar >= ar_upright)
        s_ar = float(np.clip((self.ar_upright - ar) / (self.ar_upright - self.ar_fall), 0.0, 1.0))

        # 2. Torso Angle: 1.0 when horizontal (angle >= angle_fall), 0.0 when upright (angle <= angle_upright)
        if has_kpts:
            s_angle = float(
                np.clip(
                    (angle_deg - self.angle_upright) / (self.angle_fall - self.angle_upright),
                    0.0,
                    1.0,
                )
            )
        else:
            s_angle = s_ar

        # 3. Floor Proximity: 1.0 when bottom_y >= ground_y, 0.0 when higher up in frame
        s_floor = float(np.clip((bottom_y - 0.50) / (self.ground_y - 0.50), 0.0, 1.0))

        # 4. Downward Velocity Spike: rapid drop
        s_vel = float(np.clip(v_down / self.vel_thresh, 0.0, 1.0))

        # Composite Kinematic Score
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

        frame_data["score"] = float(np.clip(raw_score, 0.0, 1.0))
        frame_data["details"] = (
            f"AR={ar:.2f} (s={s_ar:.2f}) | Angle={angle_deg:.1f}° (s={s_angle:.2f}) | V={v_down:.2f} | Flr={bottom_y:.2f}"
        )

        self.last_valid_result = frame_data
        return frame_data

    def segment_events(
        self, frame_scores: List[float], timestamps_s: List[float]
    ) -> List[Dict]:
        """
        Apply temporal smoothing rules to aggregate frame scores into distinct fall events.
        """
        n = len(frame_scores)
        if n == 0:
            return []

        # 1. Binary activation mask
        is_pos = [score >= self.frame_thresh for score in frame_scores]

        # 2. Extract contiguous positive segments
        raw_events: List[Dict] = []
        in_segment = False
        seg_start_idx = 0

        for i in range(n):
            if is_pos[i] and not in_segment:
                in_segment = True
                seg_start_idx = i
            elif not is_pos[i] and in_segment:
                in_segment = False
                seg_len = i - seg_start_idx
                if seg_len >= self.min_consecutive:
                    t_start = timestamps_s[seg_start_idx]
                    t_end = timestamps_s[i - 1]
                    dur = t_end - t_start
                    if dur >= self.min_duration:
                        peak_conf = float(np.max(frame_scores[seg_start_idx:i]))
                        raw_events.append({
                            "start_idx": seg_start_idx,
                            "end_idx": i - 1,
                            "start_time_seconds": t_start,
                            "end_time_seconds": t_end,
                            "confidence": peak_conf,
                        })

        if in_segment:
            seg_len = n - seg_start_idx
            if seg_len >= self.min_consecutive:
                t_start = timestamps_s[seg_start_idx]
                t_end = timestamps_s[n - 1]
                dur = t_end - t_start
                if dur >= self.min_duration:
                    peak_conf = float(np.max(frame_scores[seg_start_idx:n]))
                    raw_events.append({
                        "start_idx": seg_start_idx,
                        "end_idx": n - 1,
                        "start_time_seconds": t_start,
                        "end_time_seconds": t_end,
                        "confidence": peak_conf,
                    })

        if not raw_events:
            return []

        # 3. Merge nearby events separated by <= merge_gap_seconds
        merged_events: List[Dict] = []
        curr = raw_events[0]

        for next_ev in raw_events[1:]:
            gap = next_ev["start_time_seconds"] - curr["end_time_seconds"]
            if gap <= self.merge_gap:
                curr["end_idx"] = next_ev["end_idx"]
                curr["end_time_seconds"] = next_ev["end_time_seconds"]
                curr["confidence"] = max(curr["confidence"], next_ev["confidence"])
            else:
                merged_events.append(curr)
                curr = next_ev
        merged_events.append(curr)

        # 4. Final format with 1-based event_id
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

        return final_events

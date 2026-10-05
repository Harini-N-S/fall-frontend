# Production Deployment & Operation Manual: Real-Time Hybrid Fall Detection System

## 1. System Overview

The **Real-Time Hybrid Fall Detection System** integrates deep learning temporal modeling with biomechanical heuristic verification using **Strategy H5 (ML Early Trigger + Heuristic Confirmation)**.

By coupling a frozen 2-layer Recurrent Neural Network ([`FallGRUClassifier`](file:///c:/Users/Balaji/OneDrive/Desktop/Fall_Detection_Project/scripts/model.py)) with a Refined Finite State Machine ([`RefinedFallDetector`](file:///c:/Users/Balaji/OneDrive/Desktop/Fall_Detection_Project/scripts/fall_detector_refined.py)), the system achieves:
- **High Precision & False-Alarm Rejection**: 100% suppression of isolated machine learning false alarms (17/17 false alarms rejected across 36 production recordings).
- **Advance Warning Lead Time**: Alerts are timestamped to the initial dynamic collapse (**mean advance lead time: +1.154 seconds** before ground impact).
- **Static Lying Suppression**: 0 duplicate alerts during post-fall motionless floor recovery.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    A[RGB Video / ROS 2 Image Topic] --> B[Stage 1: YOLOv8n-Pose\n17 Keypoints + Bounding Box]
    B --> C[Stage 2: Kinematic Analysis\nAspect Ratio, Torso Angle, v_down, bottom_y]
    C --> D[Unified 62-Dimensional Feature Vector]
    
    D --> E[Rolling 32-Frame Feature Buffer\nStride = 8 frames]
    E --> F[Stage 3: Frozen FallGRUClassifier\nThreshold = 0.50, M = 2 windows]
    F --> G{Provisional ML Trigger?}
    
    C --> H[Stage 4: Refined Heuristic FSM\nNORMAL -> FALLING -> LYING_DOWN]
    H --> I[FSM Fall Event Transition]
    
    G -- Yes (t_trigger) --> J[Stage 5: Strategy H5 Confirmation Gate\nConfirmation Window W = 1.0s]
    I --> J
    
    J -- Transition within [t_trigger, t_end + W] --> K[CONFIRMED FALL ALERT\nPreserves t_trigger Advance Timestamp]
    J -- Timeout (No FSM Transition) --> L[REJECTED ML FALSE ALARM\nSuppressed, No Public Alert]
    K --> M[ROS 2 Topic: /fall_detection/alerts\nJSON Payload Broadcast]
```

---

## 3. Hardware Requirements & Performance Profile

Benchmarked on AMD64 Architecture (Windows 11 / Python 3.14 / PyTorch 2.11):

| Pipeline Component | CPU Latency (Mean) | CPU Throughput | GPU / TensorRT (Projected) | Bottleneck Contribution |
|---|---|---|---|---|
| **Stage 1: YOLOv8n-Pose** | 139.99 ms / frame | 7.1 FPS | ~3 – 6 ms / frame (>150 FPS) | 93.1% of CPU runtime |
| **Stage 2: 62-d Feature Extraction** | 0.059 ms / frame | 16,949 vectors/s | < 0.02 ms / frame | < 0.1% |
| **Stage 3: Frozen FallGRU Forward Pass** | 7.564 ms / window | 132.2 windows/s | ~0.5 ms / window | 5.0% |
| **Stage 4: Strategy H5 Confirmation & FSM** | 1.003 ms / frame | 997.0 steps/s | < 0.1 ms / frame | 0.7% |
| **End-to-End Pipeline (Image to Alert)** | **150.28 ms / frame** | **6.65 FPS** | **> 100 FPS** | 100% |
| **Feature Stream Replay (No Pose Ext)** | **0.83 ms / frame** | **1,199.0 FPS** | - | - |

> [!NOTE]
> The custom feature engineering, recurrent GRU inference, and Strategy H5 confirmation gate together require only **8.6 ms per frame** on a standard CPU. When deployed on edge devices with GPU/NPU acceleration (e.g., NVIDIA Jetson Orin), the entire end-to-end pipeline operates comfortably at >60 FPS.

### Resource Footprint
- **Process Memory (RSS)**: ~372 MB RAM
- **Model Checkpoint Size**: 712 KB ([`models/best_fall_gru.pt`](file:///c:/Users/Balaji/OneDrive/Desktop/Fall_Detection_Project/models/best_fall_gru.pt))
- **YOLOv8 Weights Size**: 6.5 MB (`yolov8n-pose.pt`)

---

## 4. Installation & Environment Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Deployment Health & Checksums
Run the automated pre-flight sanity suite:
```bash
python scripts/verify_deployment.py
```
Expected output:
```
======================================================================
HYBRID FALL DETECTION: PRODUCTION DEPLOYMENT VERIFICATION
======================================================================
[Check 1/6] Verifying Production Model Integrity (SHA-256)...
  --> PASS: Checkpoint verified (SHA-256: 38a5e9c316b04f94...)
[Check 2/6] Verifying Model Architecture & Weights Loading...
  --> PASS: Model instantiated on cpu, forward pass verified.
[Check 3/6] Verifying Configuration Completeness...
  --> PASS: Configuration syntax and required schema keys verified.
[Check 4/6] Verifying Feature Vector Construction & Safety...
  --> PASS: Feature extraction produces safe 62-d finite vector.
[Check 5/6] Verifying RealtimeHybridDetector Logic & Reset...
  --> PASS: Detector state isolation, alert suppression, and reset verified.
[Check 6/6] Verifying Rosbag Replay Decompression...
  --> PASS: Decoded and processed 5 frames successfully from 20260817_153818.db3.
======================================================================
RESULT: ALL 6 CHECKS PASSED -- SYSTEM IS READY FOR DEPLOYMENT.
======================================================================
```

---

## 5. Operations & Execution Guide

The unified runner [`scripts/run_hybrid_pipeline.py`](file:///c:/Users/Balaji/OneDrive/Desktop/Fall_Detection_Project/scripts/run_hybrid_pipeline.py) supports all operational modes.

### Mode A: Live ROS 2 Node
Subscribes to `/camera/image_raw` and broadcasts alerts to `/fall_detection/alerts`:
```bash
python scripts/run_hybrid_pipeline.py --live-ros2
```

### Mode B: Direct Rosbag Replay (.db3)
Replays any SQLite3/Zstandard `.db3` bag file with real-time rate throttling:
```bash
# Offline benchmark replay
python scripts/run_hybrid_pipeline.py --replay-bag input/20260817_153818.db3 --output-json results/alert_153818.json

# Simulated 30 FPS playback rate
python scripts/run_hybrid_pipeline.py --replay-bag input/20260828_134619.db3 --simulate-realtime
```

### Mode C: High-Speed Feature Stream Replay
Validates detection logic at ~1,200 FPS using pre-extracted feature matrices:
```bash
# Single recording replay
python scripts/run_hybrid_pipeline.py --replay-features 20260817_153818

# All 36 recordings regression evaluation
python scripts/run_hybrid_pipeline.py --replay-features all --output-json results/regression_log.json
```

### Mode D: Video File / Webcam Stream with Visual Overlay
Renders bounding boxes, keypoint skeletons, FSM state, and alert banners:
```bash
# Video file with visual overlay
python scripts/run_hybrid_pipeline.py --video path/to/video.mp4 --visualize

# Live USB Webcam (Device 0)
python scripts/run_hybrid_pipeline.py --camera 0 --visualize
```

---

## 6. ROS 2 Interface Specifications

### Subscribed Topics
| Topic Name | Message Type | Purpose |
|---|---|---|
| `/device_0/sensor_1/Color_0/image/data` (or `/camera/image_raw`) | `sensor_msgs/msg/Image` | 640x480 RGB Camera Input Stream |

### Published Topics
| Topic Name | Message Type | Purpose |
|---|---|---|
| `/fall_detection/alerts` | `std_msgs/msg/String` (JSON Payload) | Real-time Confirmed Fall Alert Events |

### Alert Message JSON Payload Schema
```json
{
  "event_id": "FALL_20260817_153818_001",
  "session_id": "20260817_153818",
  "status": "CONFIRMED_FALL",
  "final_alert_time": 1.067,
  "ml_trigger_time": 1.067,
  "ml_end_time": 2.100,
  "heuristic_confirmation_time": 2.867,
  "advance_lead_time_seconds": 1.800,
  "confirmation_delay_seconds": 1.800,
  "event_duration_seconds": 1.800,
  "peak_ml_probability": 0.9842,
  "mean_ml_probability": 0.9125,
  "fsm_state": "LYING_DOWN"
}
```

---

## 7. Configuration Reference (`config/hybrid_detector_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model.gru_checkpoint` | `models/best_fall_gru.pt` | Path to frozen production neural weights |
| `ml_pipeline.temporal_window` | `32` | Number of history frames evaluated per window |
| `ml_pipeline.stride` | `8` | Frame cadence between consecutive GRU evaluations |
| `ml_pipeline.ml_threshold` | `0.50` | Sigmoid probability threshold for positive window |
| `ml_pipeline.postprocess_M` | `2` | Consecutive positive windows required for provisional trigger |
| `hybrid_decision.strategy` | `"H5"` | Early trigger + confirmation strategy |
| `hybrid_decision.confirmation_window_seconds` | `1.0` | Max allowable delay ($W$) for heuristic confirmation |
| `hybrid_decision.duplicate_suppression_cooldown_seconds` | `5.0` | Cooldown period after a confirmed fall |
| `hybrid_decision.suppress_during_static_lying` | `true` | Suppress new triggers while subject remains on the floor |

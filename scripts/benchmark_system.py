"""
benchmark_system.py - Production System Latency, Throughput & Resource Profiling
================================================================================
Measures latency and throughput across each individual pipeline stage and end-to-end:
  - Stage 1: YOLOv8n-pose Inference (ms/frame)
  - Stage 2: 62-d Kinematic & Pose Feature Extraction (ms/frame)
  - Stage 3: Frozen GRU Temporal Window Inference (ms/window)
  - Stage 4: Strategy H5 State Machine & Confirmation Logic (ms/frame)
  - End-to-End: Raw RGB Frame to Published Alert Decision (ms/frame, FPS)
  - Memory Footprint: Process RAM utilization
Saves results to results/deployment_benchmark.json.
"""

import os
import sys
import time
import json
import yaml
import platform
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.model import FallGRUClassifier
from scripts.hybrid_detector import RealtimeHybridDetector

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "hybrid_detector_config.yaml")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "results", "deployment_benchmark.json")


def compute_statistics(durations_ms: list) -> dict:
    arr = np.array(durations_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "median_p50_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "samples": len(durations_ms),
    }


def get_process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return -1.0


def benchmark_all():
    print("=" * 75)
    print("HYBRID FALL DETECTION: COMPONENT & END-TO-END BENCHMARKING SUITE")
    print("=" * 75)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Platform: {platform.platform()} | Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__} | Device: {device}")
    
    # Instantiate detector
    detector = RealtimeHybridDetector(CONFIG_PATH)
    
    # -------------------------------------------------------------
    # Stage 1: YOLOv8n-pose Inference
    # -------------------------------------------------------------
    print("\n[Stage 1/5] Profiling YOLOv8n-Pose Inference...")
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = detector.heuristic_detector.model(dummy_img, verbose=False)
    
    yolo_times = []
    n_yolo = 50
    for _ in range(n_yolo):
        t0 = time.perf_counter()
        _ = detector.heuristic_detector.model(dummy_img, verbose=False)
        t1 = time.perf_counter()
        yolo_times.append((t1 - t0) * 1000.0)
    
    yolo_stats = compute_statistics(yolo_times)
    yolo_stats["fps"] = round(1000.0 / yolo_stats["mean_ms"], 1)
    print(f"  --> Mean: {yolo_stats['mean_ms']} ms | p50: {yolo_stats['median_p50_ms']} ms | p95: {yolo_stats['p95_ms']} ms | Throughput: {yolo_stats['fps']} FPS")

    # -------------------------------------------------------------
    # Stage 2: 62-d Kinematic Feature Vector Extraction
    # -------------------------------------------------------------
    print("\n[Stage 2/5] Profiling 62-d Feature Extraction...")
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.ones(17, dtype=np.float32) * 0.9
    kin_dict = {
        "box": [100.0, 50.0, 300.0, 400.0],
        "keypoints": kpts,
        "keypoints_conf": confs,
        "ar": 1.4,
        "angle_deg": 12.0,
        "v_down": 0.05,
        "bottom_y": 0.55,
        "has_person": True
    }
    
    feat_times = []
    n_feat = 1000
    for _ in range(n_feat):
        t0 = time.perf_counter()
        _ = detector.extract_62_features(dummy_img, kin_dict)
        t1 = time.perf_counter()
        feat_times.append((t1 - t0) * 1000.0)
    
    feat_stats = compute_statistics(feat_times)
    feat_stats["throughput_vectors_per_sec"] = round(1000.0 / feat_stats["mean_ms"], 1)
    print(f"  --> Mean: {feat_stats['mean_ms']} ms | p50: {feat_stats['median_p50_ms']} ms | p95: {feat_stats['p95_ms']} ms | Throughput: {feat_stats['throughput_vectors_per_sec']} vecs/s")

    # -------------------------------------------------------------
    # Stage 3: Frozen FallGRUClassifier Inference (32-frame window)
    # -------------------------------------------------------------
    print("\n[Stage 3/5] Profiling Frozen FallGRUClassifier Forward Pass...")
    gru_model = detector.gru_model
    gru_model.eval()
    dummy_window = torch.randn(1, 32, 62, dtype=torch.float32, device=detector.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = gru_model(dummy_window)
    
    gru_times = []
    n_gru = 1000
    with torch.no_grad():
        for _ in range(n_gru):
            t0 = time.perf_counter()
            _ = gru_model(dummy_window)
            t1 = time.perf_counter()
            gru_times.append((t1 - t0) * 1000.0)
    
    gru_stats = compute_statistics(gru_times)
    gru_stats["throughput_windows_per_sec"] = round(1000.0 / gru_stats["mean_ms"], 1)
    print(f"  --> Mean: {gru_stats['mean_ms']} ms | p50: {gru_stats['median_p50_ms']} ms | p95: {gru_stats['p95_ms']} ms | Throughput: {gru_stats['throughput_windows_per_sec']} windows/s")

    # -------------------------------------------------------------
    # Stage 4: Strategy H5 State Machine & Confirmation Logic
    # -------------------------------------------------------------
    print("\n[Stage 4/5] Profiling Strategy H5 FSM & Confirmation Step...")
    dummy_feat = np.zeros(62, dtype=np.float32)
    kin_step_dict = {
        "timestamp_s": 1.0,
        "has_person": True,
        "ar": 1.5,
        "angle_deg": 10.0,
        "v_down": 0.0,
        "bottom_y": 0.4,
        "is_upright": True,
        "is_horizontal": False,
        "raw_score": 0.05
    }
    detector.reset("benchmark_session")
    
    # Pre-fill buffer so ML executes on stride
    for i in range(31):
        detector.process_feature_step(dummy_feat, kin_step_dict, timestamp_s=i * 0.033, frame_number=i)
    
    fsm_times = []
    n_fsm = 1000
    for i in range(n_fsm):
        t0 = time.perf_counter()
        _ = detector.process_feature_step(dummy_feat, kin_step_dict, timestamp_s=(32 + i) * 0.033, frame_number=32 + i)
        t1 = time.perf_counter()
        fsm_times.append((t1 - t0) * 1000.0)
    
    fsm_stats = compute_statistics(fsm_times)
    fsm_stats["throughput_steps_per_sec"] = round(1000.0 / fsm_stats["mean_ms"], 1)
    print(f"  --> Mean: {fsm_stats['mean_ms']} ms | p50: {fsm_stats['median_p50_ms']} ms | p95: {fsm_stats['p95_ms']} ms | Throughput: {fsm_stats['throughput_steps_per_sec']} steps/s")

    # -------------------------------------------------------------
    # Stage 5: End-to-End Real-Time Pipeline (Raw Image -> Alert)
    # -------------------------------------------------------------
    print("\n[Stage 5/5] Profiling End-to-End Real-Time Pipeline (Image -> Pose -> Features -> GRU -> FSM -> Alert)...")
    detector.reset("e2e_benchmark")
    
    # Warmup
    for i in range(5):
        _ = detector.process_frame(dummy_img, timestamp_s=i * 0.033, frame_number=i)
    
    e2e_times = []
    n_e2e = 50
    for i in range(n_e2e):
        t0 = time.perf_counter()
        _ = detector.process_frame(dummy_img, timestamp_s=(10 + i) * 0.033, frame_number=10 + i)
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000.0)
    
    e2e_stats = compute_statistics(e2e_times)
    e2e_stats["pipeline_fps"] = round(1000.0 / e2e_stats["mean_ms"], 2)
    print(f"  --> Mean: {e2e_stats['mean_ms']} ms | p50: {e2e_stats['median_p50_ms']} ms | p95: {e2e_stats['p95_ms']} ms | Overall Pipeline: {e2e_stats['pipeline_fps']} FPS")

    # Memory
    ram_mb = get_process_memory_mb()
    print(f"\n[Resource Footprint] Process RAM (RSS): {ram_mb} MB")

    # Compile Benchmark Report
    benchmark_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": device,
            "process_ram_rss_mb": ram_mb,
        },
        "stage_1_yolo_pose": yolo_stats,
        "stage_2_feature_extraction": feat_stats,
        "stage_3_gru_inference": gru_stats,
        "stage_4_fsm_strategy_h5": fsm_stats,
        "end_to_end_pipeline": e2e_stats,
        "realtime_budget": {
            "target_camera_fps": 30.0,
            "frame_budget_ms": 33.33,
            "actual_mean_latency_ms": e2e_stats["mean_ms"],
            "margin_ms": round(33.33 - e2e_stats["mean_ms"], 2),
            "meets_realtime_30fps_target": bool(e2e_stats["pipeline_fps"] >= 30.0),
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    
    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETED: Structured report saved to {OUTPUT_JSON}")
    print("=" * 75)
    return benchmark_report


if __name__ == "__main__":
    benchmark_all()

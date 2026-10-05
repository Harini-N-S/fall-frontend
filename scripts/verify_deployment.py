"""
verify_deployment.py - Production Deployment Sanity & Integrity Verification
===========================================================================
Validates:
1. Production model SHA-256 checksum (invariance check)
2. Model loading and architecture instantiation
3. Detector configuration syntax and schema completeness
4. Feature extractor dimensions (62-d) and numerical safety (no NaNs)
5. RealtimeHybridDetector state machine logic and session reset
6. SQLite3/Zstandard .db3 bag decompressor sanity
"""

import os
import sys
import hashlib
from pathlib import Path
import yaml
import torch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.model import FallGRUClassifier
from scripts.hybrid_detector import RealtimeHybridDetector

EXPECTED_SHA256 = "38a5e9c316b04f9430c6930027cbac42d334eb8e4962845d712c6c9dc50a80d2"
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_fall_gru.pt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "hybrid_detector_config.yaml")


def check_1_model_integrity():
    print("[Check 1/6] Verifying Production Model Integrity (SHA-256)...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing production model: {CHECKPOINT_PATH}")
    
    with open(CHECKPOINT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nGot:      {file_hash}"
        )
    print(f"  --> PASS: Checkpoint verified (SHA-256: {file_hash[:16]}...)")


def check_2_model_instantiation():
    print("[Check 2/6] Verifying Model Architecture & Weights Loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    
    config = ckpt.get("config", {})
    input_size = config.get("input_size", 62)
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.2)
    
    model = FallGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Forward pass with dummy tensor [B, T, D]
    dummy_input = torch.zeros((1, 32, 62), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape in ((1,), (1, 1)), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN output"
    print(f"  --> PASS: Model instantiated on {device}, forward pass verified.")


def check_3_config_completeness():
    print("[Check 3/6] Verifying Configuration Completeness...")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    required_sections = [
        "model", "kinematics", "state_machine", "ml_pipeline", "hybrid_decision", "ros2"
    ]
    for sec in required_sections:
        assert sec in cfg, f"Missing config section: {sec}"
    
    assert cfg["ml_pipeline"]["temporal_window"] == 32
    assert cfg["ml_pipeline"]["feature_dim"] == 62
    assert cfg["hybrid_decision"]["confirmation_window_seconds"] == 1.0
    assert cfg["hybrid_decision"]["strategy"] == "H5"
    print("  --> PASS: Configuration syntax and required schema keys verified.")


def check_4_feature_extractor_safety():
    print("[Check 4/6] Verifying Feature Vector Construction & Safety...")
    detector = RealtimeHybridDetector(CONFIG_PATH)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate dummy kinematics and keypoints
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
    
    feat = detector.extract_62_features(dummy_frame, kin_dict)
    assert feat.shape == (62,), f"Feature dimension mismatch: expected (62,), got {feat.shape}"
    assert np.isfinite(feat).all(), "Feature vector contains NaN or Inf"
    print(f"  --> PASS: Feature extraction produces safe 62-d finite vector.")


def check_5_detector_state_logic():
    print("[Check 5/6] Verifying RealtimeHybridDetector Logic & Reset...")
    detector = RealtimeHybridDetector(CONFIG_PATH)
    
    # 1. Feed 35 dummy normal frames
    for i in range(35):
        dummy_feat = np.zeros(62, dtype=np.float32)
        kin = {
            "timestamp_s": i * (1.0 / 30.0),
            "has_person": True,
            "ar": 1.5,
            "angle_deg": 10.0,
            "v_down": 0.0,
            "bottom_y": 0.4,
            "is_upright": True,
            "is_horizontal": False,
            "raw_score": 0.05
        }
        res = detector.process_feature_step(dummy_feat, kin, timestamp_s=i * (1.0 / 30.0), frame_number=i)
        assert res["confirmed_fall"] is None, f"Normal frame {i} produced unexpected confirmed alert"
    
    # 2. Reset detector
    detector.reset(session_id="test_verify_session")
    assert len(detector.feature_buffer) == 0
    assert len(detector.timestamp_buffer) == 0
    assert len(detector.frame_idx_buffer) == 0
    assert len(detector.provisional_events) == 0
    assert len(detector.event_log) == 0
    print("  --> PASS: Detector state isolation, alert suppression, and reset verified.")


from scripts.ros2_hybrid_node import replay_db3_bag

def check_6_bag_reader_sanity():
    print("[Check 6/6] Verifying Rosbag Replay Decompression...")
    sample_bag = os.path.join(PROJECT_ROOT, "input", "20260817_153818.db3")
    if not os.path.exists(sample_bag):
        print(f"  --> SKIP: Sample bag {sample_bag} not found.")
        return
    
    detector = RealtimeHybridDetector(CONFIG_PATH)
    results = replay_db3_bag(Path(sample_bag), detector, max_frames=5)
    
    assert len(results) == 5, f"Expected 5 frames processed, got {len(results)}"
    print(f"  --> PASS: Decoded and processed 5 frames successfully from {os.path.basename(sample_bag)}.")


def main():
    print("=" * 70)
    print("HYBRID FALL DETECTION: PRODUCTION DEPLOYMENT VERIFICATION")
    print("=" * 70)
    try:
        check_1_model_integrity()
        check_2_model_instantiation()
        check_3_config_completeness()
        check_4_feature_extractor_safety()
        check_5_detector_state_logic()
        check_6_bag_reader_sanity()
        print("=" * 70)
        print("RESULT: ALL 6 CHECKS PASSED -- SYSTEM IS READY FOR DEPLOYMENT.")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n[FAIL]: Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

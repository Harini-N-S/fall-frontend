#!/usr/bin/env python3
"""
model.py - GRU-Based Temporal Sequence Binary Classifier for Fall Detection.

Architecture:
    Input:
        x of shape [B, T, 62] (temporal window of 62 kinematic & pose features)
    Temporal Encoder:
        2-Layer GRU (input_size=62, hidden_size=128, dropout=0.2, batch_first=True)
    Classifier Head:
        Linear(128 -> 64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 1)
    Output:
        Logits of shape [B] (unnormalized log-odds for BCEWithLogitsLoss)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project scripts to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))


class FallGRUClassifier(nn.Module):
    """
    Bidirectional/Unidirectional GRU sequence binary classifier for fall detection.
    """

    def __init__(
        self,
        input_size: int = 62,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: Feature dimension per frame (default: 62).
            hidden_size: Hidden state dimension of GRU (default: 128).
            num_layers: Number of stacked GRU layers (default: 2).
            dropout: Dropout probability between GRU layers and in classifier head (default: 0.2).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Recurrent temporal encoder
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape [B, T, input_size]

        Returns:
            logits: Tensor of shape [B] (unnormalized logits, no sigmoid applied)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, T, D], got {x.dim()}D tensor with shape {x.shape}")
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.size(-1)}")

        # out: [B, T, hidden_size]
        out, _ = self.gru(x)

        # Extract final time-step hidden representation
        final_timestep = out[:, -1, :]  # [B, hidden_size]

        # Feedforward classification head: [B, hidden_size] -> [B, 1]
        logits = self.classifier(final_timestep)

        # Squeeze trailing singleton dimension: [B, 1] -> [B]
        return logits.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    input_size: int = 62,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
) -> FallGRUClassifier:
    """
    Factory helper to instantiate and move model to appropriate compute device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FallGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.to(device)
    return model


@torch.no_grad()
def predict_probability(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Computes fall probabilities in [0.0, 1.0] by applying sigmoid ONLY at inference.

    Args:
        model: FallGRUClassifier model
        x: Input tensor of shape [B, T, 62]

    Returns:
        probabilities: Tensor of shape [B] with values in [0.0, 1.0]
    """
    model.eval()
    logits = model(x)
    return torch.sigmoid(logits)


def run_smoke_test():
    """
    Executes smoke test on dummy input and test pass on real dataset DataLoader.
    """
    print("=" * 80)
    print("       TESTING FALL GRU CLASSIFIER ARCHITECTURE (STEP 5)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    # 1. Instantiate Model
    model = create_model(
        input_size=62,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        device=device,
    )
    total_params = count_parameters(model)
    print(f"\nModel Architecture:\n{model}")
    print(f"\nTotal Trainable Parameters: {total_params:,}")

    # 2. Dummy Input Smoke Test
    print("\n--- Running Dummy Input Smoke Test ---")
    batch_size = 4
    seq_len = 32
    feat_dim = 62

    dummy_input = torch.randn(batch_size, seq_len, feat_dim, device=device)
    print(f"Dummy input shape:  {list(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probs = predict_probability(model, dummy_input)

    print(f"Output logits shape: {list(logits.shape)}")
    print(f"Output logits sample: {[round(v, 4) for v in logits.tolist()]}")
    print(f"Output probs shape:  {list(probs.shape)}")
    print(f"Output probs sample:  {[round(v, 4) for v in probs.tolist()]}")

    # Assertions
    assert logits.shape == (batch_size,), f"Expected logits shape ({batch_size},), got {logits.shape}"
    assert probs.shape == (batch_size,), f"Expected probs shape ({batch_size},), got {probs.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isinf(logits).any(), "Logits contain Inf!"
    assert not torch.isnan(probs).any(), "Probs contain NaN!"
    assert not torch.isinf(probs).any(), "Probs contain Inf!"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probs not in [0, 1]!"
    print("[PASS] Dummy input smoke test passed successfully!")

    # 3. Real Dataset Batch Test
    print("\n--- Running Real Dataset Batch Test ---")
    try:
        from dataset import FallTemporalDataset

        train_ds = FallTemporalDataset(
            split="train",
            window_size=32,
            stride=8,
            fall_overlap_threshold=0.25,
        )
        print(f"Loaded Train Dataset: {len(train_ds):,} windows across {len(train_ds.recordings)} recordings")

        loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        real_x = batch["features"].to(device)  # [4, 32, 62]
        real_y = batch["label"].to(device)     # [4]

        print(f"Real batch input shape:  {list(real_x.shape)}")
        print(f"Real batch labels:       {real_y.tolist()}")

        with torch.no_grad():
            real_logits = model(real_x)
            real_probs = predict_probability(model, real_x)

        print(f"Real batch logits shape: {list(real_logits.shape)}")
        print(f"Real batch logits:       {[round(v, 4) for v in real_logits.tolist()]}")
        print(f"Real batch probs:        {[round(v, 4) for v in real_probs.tolist()]}")

        assert real_logits.shape == (4,), f"Expected shape (4,), got {real_logits.shape}"
        assert not torch.isnan(real_logits).any(), "Real batch logits contain NaN!"
        assert not torch.isinf(real_logits).any(), "Real batch logits contain Inf!"
        assert (real_probs >= 0.0).all() and (real_probs <= 1.0).all(), "Real probs not in [0, 1]!"
        print("[PASS] Real dataset batch test passed successfully!")

    except Exception as e:
        print(f"[FAIL] Error during real dataset test: {e}")
        raise e

    print("\n" + "=" * 80)
    print("            ALL ARCHITECTURE TESTS PASSED WITH 0 ERRORS")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()

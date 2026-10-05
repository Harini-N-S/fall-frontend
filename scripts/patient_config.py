"""
patient_config.py - Configurable Patient and Room Mapping for Fall Monitoring.
Decoupled from detection algorithms.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Configurable mapping from recording/video session ID to patient & room
PATIENT_CONFIG: Dict[str, Dict[str, str]] = {
    "20260828_133415": {
        "patient_id": "Patient 104",
        "room": "Ward A - Bed 12",
    },
    "20260817_153818": {
        "patient_id": "Patient 102",
        "room": "Ward B - Bed 04",
    },
    "20260817_153929": {
        "patient_id": "Patient 105",
        "room": "Ward A - Bed 08",
    },
    "20260817_160451": {
        "patient_id": "Patient 108",
        "room": "Ward C - Bed 01",
    },
    "20260817_160620": {
        "patient_id": "Patient 110",
        "room": "Ward B - Bed 06",
    },
    "20260817_160813": {
        "patient_id": "Patient 112",
        "room": "Ward A - Bed 03",
    },
    "20260817_161024": {
        "patient_id": "Patient 115",
        "room": "ICU Stepdown - Bed 07",
    },
}


def get_patient_info(session_id: Optional[str]) -> Dict[str, str]:
    """
    Returns patient ID and room bed for a given recording/session ID.
    If unmapped, returns safe prototype fallback.
    """
    if not session_id:
        return {"patient_id": "Demo Patient", "room": "Demo Bed"}

    stem = Path(session_id).stem
    if stem in PATIENT_CONFIG:
        return PATIENT_CONFIG[stem]

    return {
        "patient_id": f"Patient ({stem})",
        "room": "Ward A - General Bed",
    }

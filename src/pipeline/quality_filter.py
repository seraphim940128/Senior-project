"""Landmark quality gating.

Drops frames whose required landmarks fail the visibility / presence
thresholds, so a low-confidence frame never reaches an evaluator.
"""
from __future__ import annotations

from models import ActionLabel, Landmark, PoseFrame
from pipeline.angle_extractor import required_landmarks


def available_landmarks(
    frame: PoseFrame,
    min_visibility: float,
    min_presence: float,
    basis: str,
) -> dict[str, Landmark] | None:
    """Landmark coordinates that pass the quality gate, in the requested basis.

    Returns ``None`` when 2D image landmarks are requested but unavailable.
    """
    if basis == "2d":
        return frame.available_image_coordinates(min_visibility, min_presence)
    return frame.available_coordinates(min_visibility, min_presence)


def required_action_landmarks_present(
    action: ActionLabel,
    landmarks: dict[str, Landmark],
) -> bool:
    """True when every landmark the action's angle needs is present."""
    return all(name in landmarks for name in required_landmarks(action))

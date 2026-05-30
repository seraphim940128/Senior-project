"""Action-aware joint-angle extraction.

All angles are plain three-point angles in the camera / MediaPipe world
coordinate frame — no torso coordinate system.
"""
from __future__ import annotations

import math

from models import ActionLabel, Landmark


# ---------------------------------------------------------------------------
# primitive angle math
# ---------------------------------------------------------------------------

def vector_angle_3d(
    v1: tuple[float, float, float],
    v2: tuple[float, float, float],
) -> float:
    """Angle (degrees) between two vectors via arccos of the normalized dot."""
    norm1 = math.sqrt(sum(value * value for value in v1))
    norm2 = math.sqrt(sum(value * value for value in v2))
    if norm1 <= 1e-6 or norm2 <= 1e-6:
        return 0.0
    dot = sum(left * right for left, right in zip(v1, v2))
    cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cosine))


def angle_degrees(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Three-point angle at vertex ``b`` (3D)."""
    ab = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    cb = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    return vector_angle_3d(ab, cb)


def angle_degrees_2d(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Three-point angle at vertex ``b`` using x/y only (z forced to 0)."""
    ab = (a[0] - b[0], a[1] - b[1], 0.0)
    cb = (c[0] - b[0], c[1] - b[1], 0.0)
    return vector_angle_3d(ab, cb)


# ---------------------------------------------------------------------------
# action -> landmark triplet
# ---------------------------------------------------------------------------

# Single-channel actions. shoulder_forward_elevation is bilateral and handled
# separately. flexion and abduction share the same triplet because the 3D
# three-point vertex angle (hip-shoulder-elbow) is rotation-invariant around
# the trunk axis: pure flexion (arm forward), pure abduction (arm lateral)
# and any combination produce the *same* scalar "arm-from-trunk elevation
# angle". Since the action label is selected up-front, the geometry need not
# distinguish them - both map onto one unified elevation metric whose
# reference values live in data/reference/action_references.yaml.
#
# NOTE — 2D kinematics basis (pose.basis="2d") is the production default for a
# fixed front-facing camera. The x/y-only projection is valid for:
#   • shoulder abduction (arm moves laterally in the image plane) — accurate.
#   • elbow flexion (arm stays in the frontal plane) — accurate.
# For forward-raise actions (shoulder_flexion_*, shoulder_forward_elevation) the
# arm moves toward the camera, so front-view 2D under-measures the angle.
# This is accepted: the same unified triplet is kept, thresholds are re-tuned
# to measured 2D values, and the limitation is documented in known_limitations.
# Compensation stays 3D (pose.compensation_basis="3d") and is unaffected.
_ACTION_TRIPLETS: dict[ActionLabel, tuple[str, str, str]] = {
    ActionLabel.ELBOW_FLEXION_LEFT: ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ActionLabel.ELBOW_FLEXION_RIGHT: ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ActionLabel.SHOULDER_FLEXION_LEFT: ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    ActionLabel.SHOULDER_FLEXION_RIGHT: ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ActionLabel.SHOULDER_ABDUCTION_LEFT: ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    ActionLabel.SHOULDER_ABDUCTION_RIGHT: ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
}

# Elbow flexion inverts (max(0, 180 - theta)) so ROM grows as the arm flexes —
# i.e. for every action a larger primary angle means "more motion done".
_ELBOW_ACTIONS = {ActionLabel.ELBOW_FLEXION_LEFT, ActionLabel.ELBOW_FLEXION_RIGHT}

_FORWARD_ELEVATION_LEFT = ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW")
_FORWARD_ELEVATION_RIGHT = ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW")


def required_landmarks(action: ActionLabel) -> tuple[str, ...]:
    """Landmark names this action's angle computation needs."""
    if action == ActionLabel.SHOULDER_FORWARD_ELEVATION:
        return _FORWARD_ELEVATION_LEFT + _FORWARD_ELEVATION_RIGHT
    return _ACTION_TRIPLETS[action]


def measured_joints(action: ActionLabel) -> tuple[str, ...]:
    """Vertex landmark name(s) of the action's angle triplet(s) — the joint(s)
    the live overlay labels with the measured angle.

    ``SHOULDER_FORWARD_ELEVATION`` is bilateral (both shoulders); every other
    action has a single vertex. Pure lookup over the existing triplet tables.
    """
    if action == ActionLabel.SHOULDER_FORWARD_ELEVATION:
        return (_FORWARD_ELEVATION_LEFT[1], _FORWARD_ELEVATION_RIGHT[1])
    return (_ACTION_TRIPLETS[action][1],)


def _triplet_angle(
    landmarks: dict[str, Landmark],
    triplet: tuple[str, str, str],
    angle_fn,
) -> float | None:
    pts = [landmarks.get(name) for name in triplet]
    if any(point is None for point in pts):
        return None
    return angle_fn(pts[0], pts[1], pts[2])


def bilateral_shoulder_angles(
    landmarks: dict[str, Landmark],
    basis: str = "3d",
) -> tuple[float | None, float | None]:
    """Left and right shoulder-flexion angles (HIP-SHOULDER-ELBOW). Used by the
    symmetry evaluator. Either side is ``None`` if its landmarks are missing.
    """
    angle_fn = angle_degrees_2d if basis == "2d" else angle_degrees
    left = _triplet_angle(landmarks, _FORWARD_ELEVATION_LEFT, angle_fn)
    right = _triplet_angle(landmarks, _FORWARD_ELEVATION_RIGHT, angle_fn)
    return left, right


def primary_angle(
    action: ActionLabel,
    landmarks: dict[str, Landmark],
    basis: str = "3d",
) -> float | None:
    """The single tracked angle for ``action``. ``None`` if landmarks missing.

    - elbow flexion: ``max(0, 180 - theta)`` so flexion increases the value.
    - shoulder forward elevation: ``max(left, right)`` — bilateral, see below.
    - other actions: the raw HIP-SHOULDER-ELBOW angle.

    Forward-elevation uses ``max(L, R)`` (not ``min``) so that the rep window
    starts as soon as **either** arm crosses neutral. This puts a leading arm's
    solo rise INSIDE the rep window — the symmetry evaluator can then read
    left/right onsets directly from ``rep.samples`` instead of needing a
    pre-rep rolling buffer.
    
    ROM consequence: ``rep.dynamic_rom`` now measures the stronger arm's
    travel (``peak_max - baseline_max``), not the weaker one. A unilateral
    motion (e.g., L=140 deg, R=20 deg) passes ROM but is caught by the
    symmetry channel via LSI (``|L-R| / mean * 100`` >> 15%) — the rep still
    fails overall, just reported on a different channel.
    """
    angle_fn = angle_degrees_2d if basis == "2d" else angle_degrees

    if action == ActionLabel.SHOULDER_FORWARD_ELEVATION:
        left, right = bilateral_shoulder_angles(landmarks, basis)
        candidates = [value for value in (left, right) if value is not None]
        return max(candidates) if candidates else None

    angle = _triplet_angle(landmarks, _ACTION_TRIPLETS[action], angle_fn)
    if angle is None:
        return None
    if action in _ELBOW_ACTIONS:
        return max(0.0, 180.0 - angle)
    return angle

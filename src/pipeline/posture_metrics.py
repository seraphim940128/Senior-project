"""Frontal-plane posture metrics for compensation detection.

Both metrics use landmark x/y only (depth ``z`` is irrelevant to frontal-plane
compensation) and are **orientation-agnostic** — they do NOT assume whether the
coordinate frame's y axis points up or down. That assumption is fragile:
MediaPipe's image landmarks have y pointing down, and its world landmarks also
have y pointing down — not up. Relying on a hard-coded "up" direction silently
breaks trunk tilt (the angle wraps near +/-180 deg). So instead:

- **trunk lateral tilt** uses ``atan2(dx, |dy|)``: always a small angle in
  (-90, 90) with 0 == a vertical trunk, whichever way y points.
- **shoulder elevation** is reported as the signed shoulder->hip vertical
  offset. Its sign is frame-dependent and is NOT interpreted here — the
  evaluator compares it against the calibrated resting baseline, whose own sign
  reveals which way "up" is. Measuring against the hip (not an absolute y) also
  cancels whole-body vertical drift.

These are geometric *proxies*: trunk lateral tilt approximates trunk lean from
the shoulder/hip midline, and shoulder elevation approximates scapular elevation
from a shoulder->hip offset. They are NOT true 3D scapular kinematics — adequate
for a research-grade rule-based screen, not a clinical measurement.

Pure functions, shared by the baseline collector, the per-frame live readout
and the per-rep ``CompensationEvaluator`` so the three never diverge.
"""
from __future__ import annotations

import math

from models import Landmark

LEFT_SHOULDER = "LEFT_SHOULDER"
RIGHT_SHOULDER = "RIGHT_SHOULDER"
LEFT_HIP = "LEFT_HIP"
RIGHT_HIP = "RIGHT_HIP"

# Trunk tilt needs all four; shoulder elevation needs one shoulder + same-side
# hip; shoulder width needs both shoulders.
TRUNK_LANDMARKS: tuple[str, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

_SIDE_SHOULDER: dict[str, str] = {"left": LEFT_SHOULDER, "right": RIGHT_SHOULDER}
_SIDE_HIP: dict[str, str] = {"left": LEFT_HIP, "right": RIGHT_HIP}
_EPS = 1e-9


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def trunk_lateral_tilt_deg(landmarks: dict[str, Landmark]) -> float | None:
    """Signed lateral tilt of the trunk midline from vertical, in degrees.

    ``atan2(dx, |dy|)`` keeps the result in (-90, 90) with ``0`` == a vertical
    trunk regardless of the y-axis direction. The sign follows the raw x axis;
    callers gate on the magnitude of the deviation from baseline. ``None`` if
    any of the four trunk landmarks is missing.
    """
    points = [landmarks.get(name) for name in TRUNK_LANDMARKS]
    if any(point is None for point in points):
        return None
    left_sh, right_sh, left_hip, right_hip = points  # type: ignore[misc]
    shoulder_mid = _midpoint(left_sh, right_sh)
    hip_mid = _midpoint(left_hip, right_hip)
    dx = shoulder_mid[0] - hip_mid[0]            # lateral offset
    dy = shoulder_mid[1] - hip_mid[1]            # longitudinal extent
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return None
    return math.degrees(math.atan2(dx, abs(dy)))


def shoulder_width(landmarks: dict[str, Landmark]) -> float | None:
    """Distance between the two shoulders; ``None`` if either is missing or the
    width is degenerate. Used to scale-normalize the shoulder elevation ratio.
    """
    left = landmarks.get(LEFT_SHOULDER)
    right = landmarks.get(RIGHT_SHOULDER)
    if left is None or right is None:
        return None
    width = math.hypot(left[0] - right[0], left[1] - right[1])
    return width if width > _EPS else None


def shoulder_hip_offset(landmarks: dict[str, Landmark], side: str) -> float | None:
    """Signed vertical (y) offset of the ``side`` shoulder from the same-side hip.

    The sign depends on the coordinate frame and is not interpreted here; the
    evaluator compares it against the calibrated baseline. ``None`` if the
    shoulder or its hip is missing.
    """
    shoulder = landmarks.get(_SIDE_SHOULDER[side])
    hip = landmarks.get(_SIDE_HIP[side])
    if shoulder is None or hip is None:
        return None
    return shoulder[1] - hip[1]

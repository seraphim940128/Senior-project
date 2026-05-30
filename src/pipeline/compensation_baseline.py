"""Session-level resting-posture baseline for compensation detection.

Compensation is measured as an *offset* from the patient's own resting posture,
not against absolute geometry — this absorbs camera tilt and body-shape
differences. The baseline is captured once at session start by taking the
**median** of a fixed number of valid resting frames (median, not mean, so a
stray frame cannot drag it), and is then shared by every repetition.

Quality control: ``steady_ok`` reports whether the collected frames were steady
enough to trust. It can only check *stillness* — it cannot tell whether the
patient's resting posture was itself correct (e.g. already slouched). That is an
inherent limit of a personalized baseline and is documented for the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median, pstdev

from models import Landmark
from pipeline.posture_metrics import (
    shoulder_hip_offset,
    shoulder_width,
    trunk_lateral_tilt_deg,
)


@dataclass(frozen=True, slots=True)
class PostureBaseline:
    """Median resting posture. ``basis`` records which coordinate set it was
    built from so the evaluator can refuse a mismatched basis.

    ``*_shoulder_hip_offset`` is the resting shoulder->hip vertical offset; its
    sign tells the evaluator which way is "up", so no global y convention is
    needed.

    ``rest_landmarks`` is one representative valid resting-frame landmark dict
    (the last frame fed to the collector). The live overlay uses it to compute
    the calibrated rest primary angle on demand — needed to gate scapular-
    elevation warnings to the ascent phase of the in-flight rep.
    """

    trunk_tilt_deg: float
    shoulder_width: float
    left_shoulder_hip_offset: float
    right_shoulder_hip_offset: float
    basis: str
    frames_used: int
    rest_landmarks: dict[str, Landmark]

    def shoulder_hip_offset_for(self, side: str) -> float:
        return (
            self.left_shoulder_hip_offset
            if side == "left"
            else self.right_shoulder_hip_offset
        )


class BaselineCollector:
    """Accumulates valid resting frames, then builds a :class:`PostureBaseline`.

    A frame is "valid" only when every landmark both metrics need is present;
    invalid frames are ignored (``push`` returns ``False``) and do not count
    toward ``target``.
    """

    def __init__(
        self,
        target_frames: int,
        basis: str,
        max_trunk_tilt_spread_deg: float,
        max_shoulder_offset_spread_ratio: float,
    ) -> None:
        self._target = max(1, int(target_frames))
        self._basis = basis
        self._max_trunk_spread = float(max_trunk_tilt_spread_deg)
        self._max_shoulder_spread = float(max_shoulder_offset_spread_ratio)
        self._tilts: list[float] = []
        self._widths: list[float] = []
        self._left: list[float] = []
        self._right: list[float] = []
        self._last_landmarks: dict[str, Landmark] | None = None

    def push(self, landmarks: dict[str, Landmark]) -> bool:
        """Record one frame. Returns ``True`` if the frame was valid and used.

        Silently no-ops after ``ready`` so a caller that forgets to stop
        pushing cannot grow the internal lists without bound. Returns
        ``False`` in that case (the frame is not "used" any further).
        """
        if self.ready:
            return False
        tilt = trunk_lateral_tilt_deg(landmarks)
        width = shoulder_width(landmarks)
        left = shoulder_hip_offset(landmarks, "left")
        right = shoulder_hip_offset(landmarks, "right")
        if tilt is None or width is None or left is None or right is None:
            return False
        self._tilts.append(tilt)
        self._widths.append(width)
        self._left.append(left)
        self._right.append(right)
        # Keep a representative resting frame for ``rest_landmarks``; copy so
        # downstream callers can't mutate it.
        self._last_landmarks = dict(landmarks)
        return True

    @property
    def count(self) -> int:
        return len(self._tilts)

    @property
    def target(self) -> int:
        return self._target

    @property
    def ready(self) -> bool:
        return self.count >= self._target

    @property
    def steady_ok(self) -> bool:
        """True when the collected resting frames were steady enough to trust.

        The spread (population stdev) of trunk tilt and of the shoulder->hip
        offset must each be within the configured limit. With fewer than two
        frames there is nothing to judge, so it returns ``True`` (``build``
        still guards the empty case).
        """
        if len(self._tilts) < 2:
            return True
        width = median(self._widths) if self._widths else 0.0
        if width <= 0.0:
            return False
        trunk_spread = pstdev(self._tilts)
        left_spread = pstdev(self._left) / width
        right_spread = pstdev(self._right) / width
        return (
            trunk_spread <= self._max_trunk_spread
            and left_spread <= self._max_shoulder_spread
            and right_spread <= self._max_shoulder_spread
        )

    def build(self) -> PostureBaseline:
        """Median of the collected frames. Raises if none were valid."""
        n = len(self._tilts)
        if n == 0 or self._last_landmarks is None:
            raise ValueError("no valid frames collected for the posture baseline")
        return PostureBaseline(
            trunk_tilt_deg=median(self._tilts),
            shoulder_width=median(self._widths),
            left_shoulder_hip_offset=median(self._left),
            right_shoulder_hip_offset=median(self._right),
            basis=self._basis,
            frames_used=n,
            rest_landmarks=dict(self._last_landmarks),
        )

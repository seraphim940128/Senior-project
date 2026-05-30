"""Core data structures for Posture Correction.

Pure dataclasses, no logic — only the types this pipeline actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

class ActionLabel(str, Enum):
    ELBOW_FLEXION_LEFT = "elbow_flexion_left"
    ELBOW_FLEXION_RIGHT = "elbow_flexion_right"
    SHOULDER_FLEXION_LEFT = "shoulder_flexion_left"
    SHOULDER_FLEXION_RIGHT = "shoulder_flexion_right"
    SHOULDER_ABDUCTION_LEFT = "shoulder_abduction_left"
    SHOULDER_ABDUCTION_RIGHT = "shoulder_abduction_right"
    SHOULDER_FORWARD_ELEVATION = "shoulder_forward_elevation"

    @classmethod
    def from_value(cls, value: str) -> "ActionLabel":
        return cls(value)

    @classmethod
    def choices(cls) -> list[str]:
        return [item.value for item in cls]


ACTION_LABELS: tuple[ActionLabel, ...] = tuple(ActionLabel)


# ---------------------------------------------------------------------------
# landmarks / frames
# ---------------------------------------------------------------------------

Landmark = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrackedLandmark:
    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None

    def coordinates(self) -> Landmark:
        return (self.x, self.y, self.z)

    def is_available(self, min_visibility: float, min_presence: float) -> bool:
        if self.visibility is None or self.presence is None:
            return False
        return self.visibility >= min_visibility and self.presence >= min_presence


@dataclass(slots=True)
class PoseFrame:
    """``landmarks``: metric/world space for kinematics.

    ``display_landmarks``: normalized image x/y for overlay and 2D angles.
    """

    landmarks: dict[str, TrackedLandmark]
    display_landmarks: dict[str, TrackedLandmark] | None = None
    frame_width: int | None = None
    frame_height: int | None = None

    @classmethod
    def from_coordinates(
        cls,
        landmarks: dict[str, Landmark],
        visibility: float | None = 1.0,
        presence: float | None = 1.0,
    ) -> "PoseFrame":
        return cls(
            landmarks={
                name: TrackedLandmark(
                    x=coords[0],
                    y=coords[1],
                    z=coords[2],
                    visibility=visibility,
                    presence=presence,
                )
                for name, coords in landmarks.items()
            },
            display_landmarks=None,
        )

    def available_coordinates(
        self,
        min_visibility: float,
        min_presence: float,
    ) -> dict[str, Landmark]:
        return {
            name: landmark.coordinates()
            for name, landmark in self.landmarks.items()
            if landmark.is_available(min_visibility=min_visibility, min_presence=min_presence)
        }

    def available_image_coordinates(
        self,
        min_visibility: float,
        min_presence: float,
    ) -> dict[str, Landmark] | None:
        """Pixel-space landmarks (aspect-correct) for 2D angle computation."""
        if not self.display_landmarks:
            return None
        if self.frame_width is None or self.frame_height is None:
            return None
        if self.frame_width <= 0 or self.frame_height <= 0:
            return None
        width = float(self.frame_width)
        height = float(self.frame_height)
        return {
            name: (landmark.x * width, landmark.y * height, landmark.z * width)
            for name, landmark in self.display_landmarks.items()
            if landmark.is_available(min_visibility=min_visibility, min_presence=min_presence)
        }


# ---------------------------------------------------------------------------
# samples / repetitions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PoseSample:
    """One processed frame inside a repetition.

    ``angle_deg`` is the physics-track (median-smoothed) primary angle —
    evaluators use this. ``raw_angle_deg`` is the pre-smoothing value, kept for
    debug payloads / charts only.
    """

    timestamp: float
    angle_deg: float
    delta_deg: float
    landmarks: dict[str, Landmark]
    image_landmarks: dict[str, Landmark] | None = None
    raw_angle_deg: float | None = None


@dataclass(slots=True)
class CompletedRepetition:
    """One segmented repetition, emitted by ``PhaseTracker``.

    Flat by design - the peak-band indices reference positions inside
    ``samples`` so downstream evaluators never re-segment. The rep onset
    is always ``samples[0]`` and the rep return is always ``samples[-1]``
    by construction (``PhaseTracker`` starts the cycle at ascent-onset and
    emits the cycle at descent-end), so no explicit ``onset_idx`` /
    ``return_idx`` fields are needed.

    ``baseline_angle`` is the *armed* neutral pose - the median of the
    pre-rep neutral-confirmation frames, captured by the tracker BEFORE
    this rep's ``samples[0]``. The pre-rep neutral frames are NOT part of
    ``samples``: ``samples[0]`` is already the ascent-onset frame (above
    baseline). This is why a debug payload that wants to show "the
    baseline" must use ``baseline_angle`` directly; there is no in-cycle
    baseline window.
    """

    action: ActionLabel
    samples: list[PoseSample]
    baseline_angle: float
    peak_angle: float
    movement_direction: int          # +1 angle increases on ascent, -1 it decreases
    peak_start_idx: int              # first sample inside the peak band
    peak_end_idx: int                # last sample inside the peak band
    started_at: float
    ended_at: float

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def dynamic_rom(self) -> float:
        return abs(self.peak_angle - self.baseline_angle)

    @property
    def duration_s(self) -> float:
        return self.ended_at - self.started_at


@dataclass(slots=True)
class PhaseSegment:
    """One phase of a repetition. ``angle_at_phase_end_deg`` is the instantaneous
    angle at the segment's last sample — NOT a ROM delta.
    """

    phase_name: str                  # "ascent" | "peak_hold" | "descent"
    start_idx: int
    end_idx: int
    angle_at_phase_end_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "angle_at_phase_end_deg": round(self.angle_at_phase_end_deg, 3),
        }


# ---------------------------------------------------------------------------
# metric results
# ---------------------------------------------------------------------------

# status values a MetricResult may carry
METRIC_STATUS = Literal[
    "ok",
    "not_applicable",
    "not_implemented",
    "reference_missing",
    "unreliable",
]


@dataclass(slots=True)
class MetricResult:
    """Unified evaluator output. Every evaluator returns this shape so the
    score aggregator and the placeholder slots can be handled uniformly.

    - ``primary_value``: the headline number (ROM degrees, symmetry diff, ...).
    - ``detail``: secondary scalars, keyed by name.
    """

    name: str
    status: str
    passed: bool | None = None
    score: float | None = None       # 0-100
    primary_value: float | None = None
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "score": round(self.score, 2) if self.score is not None else None,
            "primary_value": (
                round(self.primary_value, 3) if self.primary_value is not None else None
            ),
            "detail": {k: round(v, 3) for k, v in self.detail.items()},
        }


# ---------------------------------------------------------------------------
# debug payloads — viz / diagnostics only
# ---------------------------------------------------------------------------

class DebugPayload:
    """Marker base for per-metric debug payloads.

    Debug payloads carry intermediate time-series for live charting and
    diagnostics. They MUST NOT be imported by the scoring or feedback layers —
    that isolation is enforced by ``tests/test_no_debug_in_score.py``.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class RomDebugPayload(DebugPayload):
    """ROM chart payload. ``baseline_deg`` is the rep's armed neutral pose
    (see ``CompletedRepetition.baseline_angle``) - the chart draws it as a
    horizontal reference line. No baseline *window* field: the pre-rep
    neutral frames live before ``samples[0]`` and have no index inside
    ``smoothed_angles_deg``.
    """

    raw_angles_deg: list[float]
    smoothed_angles_deg: list[float]
    timestamps_s: list[float]
    baseline_deg: float
    peak_idx: int
    threshold_min_deg: float | None      # required_min_deg, for the chart line


@dataclass(frozen=True, slots=True)
class PhaseRomDebugPayload(DebugPayload):
    smoothed_angles_deg: list[float]
    timestamps_s: list[float]
    baseline_deg: float
    peak_deg: float
    peak_band_low_deg: float             # peak - peak_tolerance_deg
    peak_start_idx: int
    peak_end_idx: int
    ascent_threshold_deg: float | None   # target_rom * phase_rom_ratio
    descent_threshold_deg: float | None


@dataclass(frozen=True, slots=True)
class CompensationDebugPayload(DebugPayload):
    """Per-frame compensation series for the live chart. All values are the
    median-smoothed, baseline-relative quantities the evaluator scored on.
    """

    timestamps_s: list[float]
    trunk_tilt_deviation_deg: list[float]      # |trunk tilt - baseline|, per frame
    shoulder_elevation_ratio_left: list[float]  # rise / baseline shoulder width
    shoulder_elevation_ratio_right: list[float]
    trunk_tilt_threshold_deg: float
    shoulder_elevation_threshold_ratio: float
    trunk_tilt_detected: bool
    shoulder_elevation_detected: bool
    basis_used: Literal["2d", "3d"]
    # Per-frame primary (arm) angle + legacy setting-phase boundary retained
    # for diagnostics. Scored shoulder-action SE detection now uses the rep's
    # ascent segment; elbow flexion remains full-rep.
    primary_angle_deg: list[float]
    rep_baseline_angle_deg: float
    setting_phase_max_deg: float
    # Full-rep sample index where the scored SE setting-phase window ends, so the
    # live chart can mark where SE detection stops counting. ``-1`` means no
    # restriction (elbow flexion / no baseline / empty rep).
    shoulder_setting_phase_end_idx: int = -1


@dataclass(frozen=True, slots=True)
class SymmetryDebugPayload(DebugPayload):
    raw_left_deg: list[float]
    raw_right_deg: list[float]
    smoothed_left_deg: list[float]
    smoothed_right_deg: list[float]
    timestamps_s: list[float]
    waveform_similarity: float       # amplitude-normalized left/right agreement, [0,1]
    left_percentile_5: float
    left_percentile_95: float
    right_percentile_5: float
    right_percentile_95: float
    left_peak_index: int
    right_peak_index: int
    basis_used: Literal["2d", "3d"]
    median_window: int
    # Onset-timing diagnostics (from rep.samples; the pre-rep buffer is legacy).
    # ``-1`` = not available (an arm never crossed the velocity threshold).
    lsi_pct: float = 0.0
    left_onset_s: float = -1.0
    right_onset_s: float = -1.0
    onset_diff_s: float = -1.0


# ---------------------------------------------------------------------------
# live overlay readouts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    """Resting-posture calibration state, surfaced to the overlay each frame.

    ``state`` is one of ``off`` (no calibration — pipeline runs immediately),
    ``awaiting`` (waiting for the user to press R), ``countdown``, ``collecting``
    or ``active`` (baseline ready, detection running). ``message`` is the text
    the overlay banner draws verbatim.
    """

    state: str
    message: str
    countdown_remaining_s: float | None = None
    frames_collected: int = 0
    frames_target: int = 0


@dataclass(frozen=True, slots=True)
class LiveCompensation:
    """Per-frame compensation readout for the live overlay (not a scored metric).

    Values are baseline-relative; ``None`` when the required landmarks for that
    channel are missing this frame. The booleans drive the on-screen warning.
    """

    trunk_tilt_deviation_deg: float | None
    shoulder_elevation_ratio: float | None      # worst patient side
    aux_shoulder_elevation_ratio: float | None  # contralateral side, advisory only
    trunk_tilt_over_threshold: bool
    shoulder_elevation_over_threshold: bool
    aux_shoulder_elevation_over_threshold: bool


@dataclass(slots=True)
class JointReadout:
    """One measured joint's angle for the live overlay.

    ``x`` / ``y`` are the joint's pixel position in the camera image;
    ``angle_2d`` / ``angle_3d`` are the same pipeline-computed angle in the
    image-plane and world bases. Pure data — the overlay only draws it.
    """

    joint: str
    x: float
    y: float
    angle_2d: float | None
    angle_3d: float | None

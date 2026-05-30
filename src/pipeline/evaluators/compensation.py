"""Compensation evaluator — rule-based trunk-lateral-shift + scapular-elevation
detection.

Compensation is measured as an *offset* from a session-start resting-posture
baseline (:class:`~pipeline.compensation_baseline.PostureBaseline`). Two
channels:

- **trunk lateral tilt (TLS)** — all shoulder actions, including bilateral
  forward elevation.
- **scapular elevation (SE)** — every action, on the patient-side shoulder(s).

Each channel's per-frame, baseline-relative series is gap-aware median-smoothed,
then *confirmed* via one of two persistence modes (selected per channel via
config, see ``_channel``):

- **peak-based** (``min_persist_seconds <= 0``): a single smoothed value over the
  threshold confirms detection — no sustained hold required.
- **time-based** (the default for both TLS ``min_persist_seconds`` and SE
  ``shoulder_elevation_min_persist_seconds``): the over-threshold run must last at
  least the configured number of seconds; FPS-independent. This reduces transient
  tracking and natural-movement false positives.

``passed`` is the confirmation gate; ``score`` is a linear ramp on the smoothed
peak so it degrades smoothly once the threshold is crossed.

**SE is restricted to the rep's setting phase for shoulder actions** — the
ascent frames whose primary angle is within
``shoulder_elevation_setting_phase_max_deg`` of the rep baseline. Past the
setting phase, scapulohumeral rhythm naturally lifts the acromion landmark; a
single landmark cannot tell that normal motion from a true shrug. Clinically,
shoulder hiking is an initiation compensation. Elbow flexion does not engage
scapular rhythm, so its SE channel is full-rep.

Missing landmarks: a frame whose landmarks are absent is a *gap* (``None``). A
gap is NOT carried forward — it breaks any persistence run, and a channel with
too few valid frames (< ``min_valid_frame_fraction``) reports ``unreliable``
rather than guessing.

Trunk lean *backward* (TLB) for forward elevation is deliberately not detected:
a single frontal camera cannot resolve sagittal lean. Frontal-plane lateral
tilt remains measurable and is checked for forward elevation.

The x/y coordinate source follows ``pose.compensation_basis`` (world coords
for ``3d``, image pixels for ``2d``), independently from kinematics metrics.

Limitations (research-grade proxy, NOT a clinical measurement tool):

- trunk tilt and scapular elevation are geometric *proxies* from shoulder/hip
  landmarks — not true 3D scapular kinematics.
- the thresholds are literature- and field-seeded values, not yet clinically
  validated against therapist scoring.
- fast reps where the ascent segment covers very few frames may return
  ``unreliable`` for the SE channel (cannot confirm persistence in a short
  window). This is the honest behavior — better than guessing.
- single-camera 2D: body rotation and occlusion degrade accuracy.
"""
from __future__ import annotations

from statistics import median

from config import CompensationSettings
from models import (
    ActionLabel,
    CompensationDebugPayload,
    CompletedRepetition,
    Landmark,
    MetricResult,
    PoseSample,
)
from pipeline.compensation_baseline import PostureBaseline
from pipeline.posture_metrics import shoulder_hip_offset, trunk_lateral_tilt_deg

# Actions whose compensation set includes trunk lateral shift. Trunk lateral
# lean can substitute for the prime mover in *every* supported action — including
# elbow flexion (leaning the torso to assist a curl) — so it is checked for all
# of them. The trunk series is always full-rep, so the lean is caught whenever it
# occurs in the cycle.
_TRUNK_TILT_ACTIONS: frozenset[ActionLabel] = frozenset(
    {
        ActionLabel.ELBOW_FLEXION_LEFT,
        ActionLabel.ELBOW_FLEXION_RIGHT,
        ActionLabel.SHOULDER_FLEXION_LEFT,
        ActionLabel.SHOULDER_FLEXION_RIGHT,
        ActionLabel.SHOULDER_ABDUCTION_LEFT,
        ActionLabel.SHOULDER_ABDUCTION_RIGHT,
        ActionLabel.SHOULDER_FORWARD_ELEVATION,
    }
)

# Shoulder actions — SE evaluation is restricted to the rep's ascent segment.
# Elbow flexion (the only non-shoulder action) is evaluated full-rep.
_SHOULDER_ACTIONS: frozenset[ActionLabel] = frozenset(
    {
        ActionLabel.SHOULDER_FLEXION_LEFT,
        ActionLabel.SHOULDER_FLEXION_RIGHT,
        ActionLabel.SHOULDER_ABDUCTION_LEFT,
        ActionLabel.SHOULDER_ABDUCTION_RIGHT,
        ActionLabel.SHOULDER_FORWARD_ELEVATION,
    }
)

# Patient-side shoulder(s) whose elevation is checked, per action.
_SHOULDER_SIDES: dict[ActionLabel, tuple[str, ...]] = {
    ActionLabel.ELBOW_FLEXION_LEFT: ("left",),
    ActionLabel.ELBOW_FLEXION_RIGHT: ("right",),
    ActionLabel.SHOULDER_FLEXION_LEFT: ("left",),
    ActionLabel.SHOULDER_FLEXION_RIGHT: ("right",),
    ActionLabel.SHOULDER_ABDUCTION_LEFT: ("left",),
    ActionLabel.SHOULDER_ABDUCTION_RIGHT: ("right",),
    ActionLabel.SHOULDER_FORWARD_ELEVATION: ("left", "right"),
}

_AUX_SHOULDER_SIDE: dict[ActionLabel, str] = {
    ActionLabel.ELBOW_FLEXION_LEFT: "right",
    ActionLabel.ELBOW_FLEXION_RIGHT: "left",
    ActionLabel.SHOULDER_FLEXION_LEFT: "right",
    ActionLabel.SHOULDER_FLEXION_RIGHT: "left",
    ActionLabel.SHOULDER_ABDUCTION_LEFT: "right",
    ActionLabel.SHOULDER_ABDUCTION_RIGHT: "left",
}


def compensation_shoulder_sides(action: ActionLabel) -> tuple[str, ...]:
    """Patient-side shoulder(s) the evaluator checks for scapular elevation.

    Public so the live overlay applies the SAME per-action rule the scored
    result uses — the warning cannot then disagree with the rep result.
    """
    return _SHOULDER_SIDES[action]


def compensation_checks_trunk(action: ActionLabel) -> bool:
    """Whether the evaluator checks trunk lateral shift for ``action``."""
    return action in _TRUNK_TILT_ACTIONS


def compensation_aux_shoulder_side(action: ActionLabel) -> str | None:
    """Contralateral shoulder observed for advisory output on unilateral actions."""
    return _AUX_SHOULDER_SIDE.get(action)


def compensation_uses_ascent_phase(action: ActionLabel) -> bool:
    """Whether the SE channel restricts to the rep's setting phase for
    ``action`` — the early ascent within ``shoulder_elevation_setting_phase_max_deg``
    of the rep baseline. True for shoulder actions (flexion / abduction / forward
    elevation), False for elbow flexion. Public so the live overlay's
    shoulder-warning gate matches the scored result.
    """
    return action in _SHOULDER_ACTIONS


def _linear_score(value: float, full_score_at: float, zero_score_at: float) -> float:
    """Map ``value`` to 0-100: ``full_score_at`` -> 100, ``zero_score_at`` -> 0,
    linear in between, clamped. Works whichever endpoint is larger.
    """
    if full_score_at == zero_score_at:
        return 100.0
    frac = (value - zero_score_at) / (full_score_at - zero_score_at)
    return max(0.0, min(100.0, 100.0 * frac))


def _smooth_with_gaps(
    values: list[float | None], window: int
) -> list[float | None]:
    """Centered sliding-median smoothing that preserves gaps.

    A ``None`` (missing-landmark frame) stays ``None``; a valid frame is the
    median of the valid values inside its window. Gaps therefore neither shift
    nor fabricate values.
    """
    if window <= 1:
        return list(values)
    half = window // 2
    n = len(values)
    out: list[float | None] = []
    for i in range(n):
        if values[i] is None:
            out.append(None)
            continue
        win = [v for v in values[max(0, i - half):i + half + 1] if v is not None]
        out.append(median(win))           # win always contains values[i]
    return out


def _longest_over_threshold_duration(
    values: list[float | None], timestamps: list[float], threshold: float
) -> float:
    """Longest time span of a run of consecutive frames that are all present
    (non-gap) and strictly over ``threshold``. A gap or an under-threshold
    frame ends the run.
    """
    longest = 0.0
    run_start: int | None = None
    for i, value in enumerate(values):
        if value is not None and value > threshold:
            if run_start is None:
                run_start = i
            longest = max(longest, timestamps[i] - timestamps[run_start])
        else:
            run_start = None
    return longest


def _fill_for_display(values: list[float | None]) -> list[float]:
    """Forward-fill gaps for chart continuity ONLY. Detection never sees this —
    it always works on the gap-preserving series.
    """
    out: list[float] = []
    last = 0.0
    for value in values:
        if value is not None:
            last = value
        out.append(last)
    return out


class _ChannelResult:
    """Outcome of evaluating one compensation channel over a rep."""

    __slots__ = ("reliable", "detected", "peak", "valid_fraction")

    def __init__(
        self, reliable: bool, detected: bool, peak: float, valid_fraction: float
    ) -> None:
        self.reliable = reliable
        self.detected = detected
        self.peak = peak
        self.valid_fraction = valid_fraction


class CompensationEvaluator:
    name = "compensation"

    def __init__(
        self,
        settings: CompensationSettings,
        basis: str,
        median_window: int,
    ) -> None:
        self.settings = settings
        self.basis = basis
        self.median_window = median_window
        # Injected by LiveSession once the session-start calibration completes.
        self.baseline: PostureBaseline | None = None

    # -- per-frame channel series --------------------------------------------

    def _sample_landmarks(self, sample: PoseSample) -> dict[str, Landmark]:
        landmarks = (
            sample.image_landmarks if self.basis == "2d" else sample.landmarks
        )
        return landmarks or {}

    def _shoulder_ratio_at(
        self, sample: PoseSample, side: str, baseline: PostureBaseline
    ) -> float | None:
        """Single-frame shoulder rise ratio against the calibrated baseline."""
        offset = shoulder_hip_offset(self._sample_landmarks(sample), side)
        if offset is None:
            return None
        base_offset = baseline.shoulder_hip_offset_for(side)
        up_dir = 1.0 if base_offset >= 0.0 else -1.0
        return (offset - base_offset) * up_dir / baseline.shoulder_width

    def _trunk_series(
        self, rep: CompletedRepetition, baseline: PostureBaseline
    ) -> tuple[list[float | None], list[float]]:
        """Gap-aware median-smoothed |trunk tilt - baseline| over the full rep."""
        raw: list[float | None] = []
        timestamps: list[float] = []
        for sample in rep.samples:
            tilt = trunk_lateral_tilt_deg(self._sample_landmarks(sample))
            raw.append(None if tilt is None else abs(tilt - baseline.trunk_tilt_deg))
            timestamps.append(sample.timestamp)
        return _smooth_with_gaps(raw, self.median_window), timestamps

    def _shoulder_series(
        self,
        rep: CompletedRepetition,
        baseline: PostureBaseline,
        side: str,
        action: ActionLabel,
    ) -> tuple[list[float | None], list[float]]:
        """Gap-aware median-smoothed shoulder rise ratio used by *detection*.

        For shoulder actions, only frames in the rep's **setting phase** are kept
        — ascent frames whose primary angle is within
        ``shoulder_elevation_setting_phase_max_deg`` of the rep baseline. Later
        frames are dropped before smoothing so the median window does not pull in
        scapulohumeral-rhythm-dominated values from higher elevation. Elbow
        flexion keeps every frame.
        """
        if action in _SHOULDER_ACTIONS:
            max_dev = self.settings.shoulder_elevation_setting_phase_max_deg
            direction = rep.movement_direction or 1
            source_samples = [
                sample
                for sample in rep.samples[:rep.peak_start_idx + 1]
                if (sample.angle_deg - rep.baseline_angle) * direction <= max_dev
            ]
        else:
            source_samples = rep.samples
        raw: list[float | None] = []
        timestamps: list[float] = []
        for sample in source_samples:
            raw.append(self._shoulder_ratio_at(sample, side, baseline))
            timestamps.append(sample.timestamp)
        return _smooth_with_gaps(raw, self.median_window), timestamps

    def _shoulder_display_series(
        self, rep: CompletedRepetition, baseline: PostureBaseline, side: str
    ) -> list[float]:
        """Full-rep shoulder rise ratio, gap-filled — for the debug chart only.

        Display always covers the full rep so the chart's x axis lines up with
        the trunk series and the primary-angle trace; detection uses
        ``_shoulder_series``, which applies the ascent-segment filter.
        """
        raw = [self._shoulder_ratio_at(s, side, baseline) for s in rep.samples]
        return _fill_for_display(_smooth_with_gaps(raw, self.median_window))

    def _setting_phase_end_idx(
        self, rep: CompletedRepetition, action: ActionLabel
    ) -> int:
        """Last full-rep sample index still inside the SE setting-phase window.

        Mirrors the ascent filter in ``_shoulder_series`` so the live chart can
        mark where scored SE detection stops counting. Returns ``-1`` for actions
        with no setting-phase restriction (elbow flexion) or an empty rep.
        """
        if action not in _SHOULDER_ACTIONS or not rep.samples:
            return -1
        max_dev = self.settings.shoulder_elevation_setting_phase_max_deg
        direction = rep.movement_direction or 1
        end = -1
        for i, sample in enumerate(rep.samples[: rep.peak_start_idx + 1]):
            if (sample.angle_deg - rep.baseline_angle) * direction <= max_dev:
                end = i
        return end

    def _channel(
        self,
        series: list[float | None],
        timestamps: list[float],
        threshold: float,
        min_persist_seconds: float,
    ) -> _ChannelResult:
        """Reliability + detection + peak for one smoothed channel series.

        ``min_persist_seconds <= 0`` means *peak-based*: a single smoothed value
        over the threshold confirms detection (no hold required). Otherwise the
        over-threshold run must span at least ``min_persist_seconds``.
        """
        total = len(series)
        valid_values = [v for v in series if v is not None]
        fraction = len(valid_values) / total if total else 0.0
        reliable = (
            len(valid_values) >= 2
            and fraction >= self.settings.min_valid_frame_fraction
        )
        if not reliable:
            return _ChannelResult(False, False, 0.0, fraction)
        peak = max(valid_values)
        if min_persist_seconds <= 0.0:
            detected = peak > threshold
        else:
            duration = _longest_over_threshold_duration(series, timestamps, threshold)
            detected = duration >= min_persist_seconds
        return _ChannelResult(True, detected, peak, fraction)

    # -- public API -----------------------------------------------------------

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        baseline = self.baseline
        if baseline is None or baseline.basis != self.basis:
            # No resting baseline -> cannot measure an offset. Do not hard-fail.
            return MetricResult(
                name=self.name, status="unreliable", passed=None, score=None
            )

        trunk_thr = self.settings.trunk_tilt_threshold_deg
        shoulder_thr = self.settings.shoulder_elevation_threshold_ratio
        shoulder_score_zero = self.settings.shoulder_elevation_score_zero_ratio

        detail: dict[str, float] = {
            "trunk_tilt_threshold_deg": trunk_thr,
            "shoulder_elevation_threshold_ratio": shoulder_thr,
            "persist_seconds_required": self.settings.min_persist_seconds,
            "shoulder_persist_seconds_required": (
                self.settings.shoulder_elevation_min_persist_seconds
            ),
        }
        channel_scores: list[float] = []
        severities: list[float] = []
        fractions: list[float] = []
        trunk_detected = False
        shoulder_detected = False
        # Every channel the action *should* assess must be reliable, otherwise
        # the whole metric is unreliable — a channel that could not be measured
        # must never be packaged as a compensation "pass".
        all_channels_reliable = True

        # -- trunk lateral tilt channel (all shoulder actions) ---------------
        if action in _TRUNK_TILT_ACTIONS:
            series, timestamps = self._trunk_series(rep, baseline)
            trunk = self._channel(
                series, timestamps, trunk_thr, self.settings.min_persist_seconds
            )
            detail["trunk_channel_reliable"] = 1.0 if trunk.reliable else 0.0
            if trunk.reliable:
                trunk_detected = trunk.detected
                fractions.append(trunk.valid_fraction)
                detail["trunk_tilt_peak_deg"] = trunk.peak
                detail["trunk_tilt_detected"] = 1.0 if trunk.detected else 0.0
                channel_scores.append(
                    _linear_score(
                        trunk.peak, trunk_thr, self.settings.trunk_tilt_score_zero_deg
                    )
                )
                severities.append(trunk.peak / trunk_thr if trunk_thr > 0 else 0.0)
            else:
                all_channels_reliable = False

        # -- scapular elevation channel (every action) -----------------------
        shoulder_peak = 0.0
        shoulder_sides_reliable = True
        for side in _SHOULDER_SIDES[action]:
            series, timestamps = self._shoulder_series(rep, baseline, side, action)
            shoulder = self._channel(
                series,
                timestamps,
                shoulder_thr,
                self.settings.shoulder_elevation_min_persist_seconds,
            )
            if not shoulder.reliable:
                shoulder_sides_reliable = False
                continue
            fractions.append(shoulder.valid_fraction)
            shoulder_peak = max(shoulder_peak, shoulder.peak)
            if shoulder.detected:
                shoulder_detected = True
        detail["shoulder_channel_reliable"] = 1.0 if shoulder_sides_reliable else 0.0
        if shoulder_sides_reliable:
            detail["shoulder_elevation_peak_ratio"] = shoulder_peak
            detail["shoulder_elevation_detected"] = 1.0 if shoulder_detected else 0.0
            channel_scores.append(
                _linear_score(shoulder_peak, shoulder_thr, shoulder_score_zero)
            )
            severities.append(shoulder_peak / shoulder_thr if shoulder_thr > 0 else 0.0)
        else:
            all_channels_reliable = False

        # -- contralateral shoulder advisory (unilateral actions only) -------
        # This channel is visible in output and feedback but deliberately does
        # not affect reliability, pass/fail, or scoring for the primary task.
        # Emitted ONLY when the primary metric is reliable: an ``unreliable``
        # result must never smuggle an advisory out through ``detail``. The
        # feedback builder and the live overlay key off ``aux_*_detected``
        # without re-checking status, so suppressing it here is the single
        # source of truth that keeps both consumers correct.
        aux_side = _AUX_SHOULDER_SIDE.get(action)
        if aux_side is not None and all_channels_reliable:
            series, timestamps = self._shoulder_series(rep, baseline, aux_side, action)
            aux = self._channel(
                series,
                timestamps,
                shoulder_thr,
                self.settings.shoulder_elevation_min_persist_seconds,
            )
            detail["aux_shoulder_channel_reliable"] = 1.0 if aux.reliable else 0.0
            if aux.reliable:
                detail["aux_shoulder_elevation_peak_ratio"] = aux.peak
                detail["aux_shoulder_elevation_detected"] = 1.0 if aux.detected else 0.0

        if fractions:
            detail["valid_frame_fraction"] = min(fractions)

        if not all_channels_reliable or not channel_scores:
            # An applicable channel could not be assessed -> unreliable. The
            # per-channel ``*_channel_reliable`` flags in ``detail`` say which.
            return MetricResult(
                name=self.name, status="unreliable", passed=None, score=None,
                detail=detail,
            )

        passed = not (trunk_detected or shoulder_detected)
        score = min(channel_scores)
        if not passed:
            score = min(score, self.settings.failed_score_cap)
        return MetricResult(
            name=self.name,
            status="ok",
            passed=passed,
            score=score,
            primary_value=max(severities) if severities else 0.0,
            detail=detail,
        )

    def debug_payload(
        self, rep: CompletedRepetition, action: ActionLabel
    ) -> CompensationDebugPayload:
        trunk_thr = self.settings.trunk_tilt_threshold_deg
        shoulder_thr = self.settings.shoulder_elevation_threshold_ratio
        setting_phase_max = self.settings.shoulder_elevation_setting_phase_max_deg
        baseline = self.baseline
        start = rep.samples[0].timestamp if rep.samples else 0.0
        timestamps_rel = [s.timestamp - start for s in rep.samples]
        primary_angles = [s.angle_deg for s in rep.samples]

        if baseline is None or baseline.basis != self.basis:
            return CompensationDebugPayload(
                timestamps_s=timestamps_rel,
                trunk_tilt_deviation_deg=[],
                shoulder_elevation_ratio_left=[],
                shoulder_elevation_ratio_right=[],
                trunk_tilt_threshold_deg=trunk_thr,
                shoulder_elevation_threshold_ratio=shoulder_thr,
                trunk_tilt_detected=False,
                shoulder_elevation_detected=False,
                basis_used=self.basis,  # type: ignore[arg-type]
                primary_angle_deg=primary_angles,
                rep_baseline_angle_deg=rep.baseline_angle,
                setting_phase_max_deg=setting_phase_max,
                shoulder_setting_phase_end_idx=-1,
            )

        trunk_persist = self.settings.min_persist_seconds
        shoulder_persist = self.settings.shoulder_elevation_min_persist_seconds

        trunk_display: list[float] = []
        trunk_detected = False
        if action in _TRUNK_TILT_ACTIONS:
            series, timestamps = self._trunk_series(rep, baseline)
            trunk_detected = self._channel(
                series, timestamps, trunk_thr, trunk_persist
            ).detected
            trunk_display = _fill_for_display(series)

        sides = _SHOULDER_SIDES[action]
        left_display: list[float] = []
        right_display: list[float] = []
        shoulder_detected = False
        # Display both sides for all actions: non-task shoulder movement is an
        # advisory channel on unilateral actions and useful diagnostic context.
        if "left" in sides or _AUX_SHOULDER_SIDE.get(action) == "left":
            detect_series, ts = self._shoulder_series(rep, baseline, "left", action)
            if (
                "left" in sides
                and self._channel(detect_series, ts, shoulder_thr, shoulder_persist).detected
            ):
                shoulder_detected = True
            left_display = self._shoulder_display_series(rep, baseline, "left")
        if "right" in sides or _AUX_SHOULDER_SIDE.get(action) == "right":
            detect_series, ts = self._shoulder_series(rep, baseline, "right", action)
            if (
                "right" in sides
                and self._channel(detect_series, ts, shoulder_thr, shoulder_persist).detected
            ):
                shoulder_detected = True
            right_display = self._shoulder_display_series(rep, baseline, "right")

        return CompensationDebugPayload(
            timestamps_s=timestamps_rel,
            trunk_tilt_deviation_deg=trunk_display,
            shoulder_elevation_ratio_left=left_display,
            shoulder_elevation_ratio_right=right_display,
            trunk_tilt_threshold_deg=trunk_thr,
            shoulder_elevation_threshold_ratio=shoulder_thr,
            trunk_tilt_detected=trunk_detected,
            shoulder_elevation_detected=shoulder_detected,
            basis_used=self.basis,  # type: ignore[arg-type]
            primary_angle_deg=primary_angles,
            rep_baseline_angle_deg=rep.baseline_angle,
            setting_phase_max_deg=setting_phase_max,
            shoulder_setting_phase_end_idx=self._setting_phase_end_idx(rep, action),
        )

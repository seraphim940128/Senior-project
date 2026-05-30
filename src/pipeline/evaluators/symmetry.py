"""Symmetry evaluator for ``SHOULDER_FORWARD_ELEVATION`` (left vs right arm).

Two gates, BOTH must pass; ``score`` is the ``min`` of the two sub-scores:

- **amplitude (LSI)** — the clinical Limb Symmetry Index
  ``|left_rom - right_rom| / mean(left_rom, right_rom) * 100``. Catches one arm
  not travelling as far as the other (Physiopedia / Wilke 2020).
- **onset timing** — ``|left_onset - right_onset|`` seconds. Each arm's movement
  onset is the first time its angular speed stays ``>=`` a velocity threshold
  for a few frames (first-derivative method, Behrens 2023). Computed directly
  on ``rep.samples`` — the rep window now contains both arms' rises (see below).

Why ``rep.samples`` suffices now: ``primary_angle(FE) = max(left, right)`` (see
``angle_extractor``), so the rep is segmented as soon as **either** arm crosses
the neutral gate. A leading arm's solo rise therefore happens INSIDE the rep
window. Earlier versions used ``min(...)`` segmentation and needed a pre-rep
rolling buffer to see the lead-in; that buffer is no longer fed by
``LiveSession`` and the ``pre_rep_window`` field remains only as an escape
hatch for unit tests / future extreme-lead handling.

If ``self.pre_rep_window`` IS set externally, the evaluator falls back to the
legacy buffer-based detection (kept for backward compatibility).

``waveform_similarity`` / ``waveform_ncc`` / ``peak_lag_frac`` / ``onset_asymmetry``
are computed and reported in ``detail`` for DIAGNOSTICS only — they do NOT gate
(non-discriminating or anti-correlated on real bilateral data).

Only when BOTH arms are static is a rep ``unreliable``; one static arm is a
genuine asymmetry and fails on LSI and on the "one onset detected, one not"
timing branch.
"""
from __future__ import annotations

import numpy as np

from config import SymmetrySettings
from models import ActionLabel, CompletedRepetition, MetricResult, SymmetryDebugPayload
from pipeline.angle_extractor import bilateral_shoulder_angles
from pipeline.smoother import physics_smooth

_VARIANCE_EPS = 1e-5


def _linear_score(value: float, full_score_at: float, zero_score_at: float) -> float:
    """Map ``value`` to 0-100: ``full_score_at`` -> 100, ``zero_score_at`` -> 0,
    linear in between, clamped. Works whichever endpoint is larger.
    """
    if full_score_at == zero_score_at:
        return 100.0
    frac = (value - zero_score_at) / (full_score_at - zero_score_at)
    return max(0.0, min(100.0, 100.0 * frac))


def _normalize_unit(values: "np.ndarray") -> "np.ndarray | None":
    """Scale to ``[0, 1]`` by the 5-95 percentile range, removing amplitude.

    ``None`` when the side has no range (static) — there is no shape to compare.
    """
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    span = high - low
    if span <= _VARIANCE_EPS:
        return None
    return np.clip((values - low) / span, 0.0, 1.0)


def _movement_onset(
    timestamps: list[float],
    angles: list[float],
    velocity_threshold: float,
    confirm_frames: int,
) -> float | None:
    """First timestamp where angular speed stays ``>= velocity_threshold`` for
    ``confirm_frames`` consecutive frames. ``None`` if the arm never does.

    First-derivative onset detection: ``(angle[i]-angle[i-1]) / dt``. ``angles``
    should already be smoothed. Returns the timestamp at the START of the run.
    """
    n = len(angles)
    if n < 2 or confirm_frames < 1:
        return None
    run = 0
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0.0:
            continue
        if (angles[i] - angles[i - 1]) / dt >= velocity_threshold:
            run += 1
            if run >= confirm_frames:
                return timestamps[i - confirm_frames + 1]
        else:
            run = 0
    return None


# -- diagnostics only (not gated) --------------------------------------------

def _waveform_agreement(left: "np.ndarray", right: "np.ndarray") -> float:
    """Amplitude-normalized trajectory agreement in ``[0, 1]`` (1 = identical
    shape). ``0`` if either side is static. DIAGNOSTIC ONLY.
    """
    left_norm = _normalize_unit(left)
    right_norm = _normalize_unit(right)
    if left_norm is None or right_norm is None:
        return 0.0
    return max(0.0, 1.0 - float(np.mean(np.abs(left_norm - right_norm))))


def _peak_lag_frac(left: "np.ndarray", right: "np.ndarray") -> float:
    """``|argmax(left) - argmax(right)|`` as a fraction of the rep length.
    DIAGNOSTIC ONLY (anti-correlated with truth on real reps — both arms hold
    the top together, so peaks coincide even when one led).
    """
    n = len(left)
    if n < 2:
        return 0.0
    return abs(int(np.argmax(left)) - int(np.argmax(right))) / (n - 1)


def _onset_asymmetry(left: "np.ndarray", right: "np.ndarray") -> float:
    """Per-arm-normalized gap at the rep's first frame. DIAGNOSTIC ONLY (noisy
    for mild leads; superseded by the rep.samples onset timing)."""
    left_norm = _normalize_unit(left)
    right_norm = _normalize_unit(right)
    if left_norm is None or right_norm is None:
        return 1.0
    return abs(float(left_norm[0]) - float(right_norm[0]))


class SymmetryEvaluator:
    name = "symmetry"

    def __init__(self, settings: SymmetrySettings, basis: str, median_window: int) -> None:
        self.settings = settings
        self.basis = basis
        self.median_window = median_window
        # Legacy escape hatch: when set externally (tests, future extreme-lead
        # fallback), onset detection runs over this buffer instead of rep.samples.
        # LiveSession no longer feeds it — see module docstring.
        self.pre_rep_window: list[tuple[float, float, float]] | None = None

    def _extract(
        self, rep: CompletedRepetition
    ) -> tuple[list[float], list[float], list[float]]:
        """Per-frame left/right shoulder angles where both sides are present."""
        raw_left: list[float] = []
        raw_right: list[float] = []
        timestamps: list[float] = []
        for sample in rep.samples:
            landmarks = sample.image_landmarks if self.basis == "2d" else sample.landmarks
            if not landmarks:
                continue
            left, right = bilateral_shoulder_angles(landmarks, self.basis)
            if left is None or right is None:
                continue
            raw_left.append(float(left))
            raw_right.append(float(right))
            timestamps.append(float(sample.timestamp))
        return raw_left, raw_right, timestamps

    def _onset_times(
        self, rep: CompletedRepetition
    ) -> tuple[float | None, float | None]:
        """Per-arm movement-onset timestamps.

        Default path (``pre_rep_window is None``): scan ``rep.samples``
        directly — with ``primary_angle = max(L,R)`` the rep window starts as
        soon as either arm crosses neutral, so both arms' onsets are inside it.

        Legacy path: when ``self.pre_rep_window`` is set externally, scan that
        buffer over ``[started_at - pre_rep_lookback_s, ended_at]`` (kept for
        existing tests / extreme-lead fallback).

        Returns ``(None, None)`` when there are too few valid frames.
        """
        if self.pre_rep_window:
            return self._onset_from_buffer(rep)
        return self._onset_from_samples(rep)

    def _onset_from_samples(
        self, rep: CompletedRepetition
    ) -> tuple[float | None, float | None]:
        raw_left, raw_right, timestamps = self._extract(rep)
        if len(raw_left) < self.settings.onset_confirm_frames + 1:
            return None, None
        lefts = physics_smooth(raw_left, self.median_window)
        rights = physics_smooth(raw_right, self.median_window)
        thr = self.settings.onset_velocity_threshold_deg_s
        confirm = self.settings.onset_confirm_frames
        return (
            _movement_onset(timestamps, lefts, thr, confirm),
            _movement_onset(timestamps, rights, thr, confirm),
        )

    def _onset_from_buffer(
        self, rep: CompletedRepetition
    ) -> tuple[float | None, float | None]:
        window = self.pre_rep_window
        if not window:
            return None, None
        lo = rep.started_at - self.settings.pre_rep_lookback_s
        hi = rep.ended_at
        pts = [(t, l, r) for (t, l, r) in window if lo <= t <= hi]
        if len(pts) < self.settings.onset_confirm_frames + 1:
            return None, None
        timestamps = [p[0] for p in pts]
        lefts = physics_smooth([p[1] for p in pts], self.median_window)
        rights = physics_smooth([p[2] for p in pts], self.median_window)
        thr = self.settings.onset_velocity_threshold_deg_s
        confirm = self.settings.onset_confirm_frames
        return (
            _movement_onset(timestamps, lefts, thr, confirm),
            _movement_onset(timestamps, rights, thr, confirm),
        )

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        if action != ActionLabel.SHOULDER_FORWARD_ELEVATION:
            return MetricResult(name=self.name, status="not_applicable", passed=None, score=None)

        raw_left, raw_right, _ = self._extract(rep)
        if len(raw_left) < 2:
            return MetricResult(name=self.name, status="unreliable", passed=None, score=None)

        left = np.asarray(physics_smooth(raw_left, self.median_window))
        right = np.asarray(physics_smooth(raw_right, self.median_window))

        left_rom = float(np.percentile(left, 95) - np.percentile(left, 5))
        right_rom = float(np.percentile(right, 95) - np.percentile(right, 5))
        amp_diff = abs(left_rom - right_rom)
        mean_rom = (left_rom + right_rom) / 2.0
        lsi = (amp_diff / mean_rom * 100.0) if mean_rom > _VARIANCE_EPS else 0.0

        # -- onset timing (lead-detector), from rep.samples (buffer = legacy) --
        left_onset, right_onset = self._onset_times(rep)
        if left_onset is not None and right_onset is not None:
            onset_diff: float | None = abs(left_onset - right_onset)
        elif left_onset is None and right_onset is None:
            onset_diff = None  # no buffer / no movement detected -> can't gate on timing
        else:
            # one arm moved, the other never crossed the velocity threshold in the
            # window -> maximal timing asymmetry; force a fail.
            onset_diff = self.settings.onset_diff_score_zero_s

        # -- diagnostics only (do NOT gate) ------------------------------------
        agreement = _waveform_agreement(left, right)
        ncc = (
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > _VARIANCE_EPS and np.std(right) > _VARIANCE_EPS
            else 0.0
        )
        peak_lag = _peak_lag_frac(left, right)
        onset_asym = _onset_asymmetry(left, right)

        detail: dict[str, float] = {
            "left_rom_deg": left_rom,
            "right_rom_deg": right_rom,
            "amplitude_rom_diff_deg": amp_diff,
            "lsi_pct": lsi,
            "onset_diff_s": -1.0 if onset_diff is None else onset_diff,
            "left_onset_s": (left_onset - rep.started_at) if left_onset is not None else -99.0,
            "right_onset_s": (right_onset - rep.started_at) if right_onset is not None else -99.0,
            # diagnostics (not gated):
            "waveform_similarity": agreement,
            "waveform_ncc": ncc,
            "peak_lag_frac": peak_lag,
            "onset_asymmetry": onset_asym,
        }
        # Only a rep where BOTH arms are static is unmeasurable.
        if left_rom <= _VARIANCE_EPS and right_rom <= _VARIANCE_EPS:
            return MetricResult(
                name=self.name, status="unreliable", passed=None, score=None,
                primary_value=lsi, detail=detail,
            )

        lsi_score = _linear_score(
            lsi, self.settings.lsi_score_best_pct, self.settings.lsi_score_zero_pct
        )
        if onset_diff is None:
            onset_score = 100.0
            onset_pass = True
        else:
            onset_score = _linear_score(
                onset_diff,
                self.settings.onset_diff_score_best_s,
                self.settings.onset_diff_score_zero_s,
            )
            onset_pass = onset_diff <= self.settings.onset_diff_max_s

        passed = lsi <= self.settings.lsi_max_pct and onset_pass
        return MetricResult(
            name=self.name,
            status="ok",
            passed=passed,
            score=min(lsi_score, onset_score),
            primary_value=lsi,
            detail=detail,
        )

    def debug_payload(
        self, rep: CompletedRepetition, action: ActionLabel
    ) -> SymmetryDebugPayload:
        raw_left, raw_right, timestamps = self._extract(rep)
        left = physics_smooth(raw_left, self.median_window)
        right = physics_smooth(raw_right, self.median_window)
        left_arr = np.asarray(left) if left else np.asarray([0.0])
        right_arr = np.asarray(right) if right else np.asarray([0.0])
        agreement = _waveform_agreement(left_arr, right_arr) if len(left) >= 2 else 0.0
        l_p5, l_p95 = float(np.percentile(left_arr, 5)), float(np.percentile(left_arr, 95))
        r_p5, r_p95 = float(np.percentile(right_arr, 5)), float(np.percentile(right_arr, 95))
        mean_rom = ((l_p95 - l_p5) + (r_p95 - r_p5)) / 2.0
        lsi = (abs((l_p95 - l_p5) - (r_p95 - r_p5)) / mean_rom * 100.0) if mean_rom > _VARIANCE_EPS else 0.0
        left_onset, right_onset = self._onset_times(rep)
        onset_diff = (
            abs(left_onset - right_onset)
            if left_onset is not None and right_onset is not None
            else -1.0
        )
        start = timestamps[0] if timestamps else 0.0
        return SymmetryDebugPayload(
            raw_left_deg=list(raw_left),
            raw_right_deg=list(raw_right),
            smoothed_left_deg=list(left),
            smoothed_right_deg=list(right),
            timestamps_s=[t - start for t in timestamps],
            waveform_similarity=agreement,
            left_percentile_5=l_p5,
            left_percentile_95=l_p95,
            right_percentile_5=r_p5,
            right_percentile_95=r_p95,
            left_peak_index=int(np.argmax(left_arr)),
            right_peak_index=int(np.argmax(right_arr)),
            basis_used=self.basis,  # type: ignore[arg-type]
            median_window=self.median_window,
            lsi_pct=lsi,
            left_onset_s=(left_onset - rep.started_at) if left_onset is not None else -99.0,
            right_onset_s=(right_onset - rep.started_at) if right_onset is not None else -99.0,
            onset_diff_s=onset_diff,
        )

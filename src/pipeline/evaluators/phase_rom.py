"""Phase-ROM evaluator: ascent / peak_hold / descent decomposition.

Uses the peak-band indices ``PhaseTracker`` already computed on the
``CompletedRepetition`` — this evaluator never re-segments. Each of the ascent
and descent phases must cover ``phase_rom_ratio`` of the target ROM to pass.
"""
from __future__ import annotations

from config import PhaseRomSettings
from models import (
    ActionLabel,
    CompletedRepetition,
    MetricResult,
    PhaseRomDebugPayload,
    PhaseSegment,
)
from pipeline.evaluators.rom import RomReference


class PhaseRomEvaluator:
    name = "phase_rom"

    def __init__(
        self,
        settings: PhaseRomSettings,
        references: dict[ActionLabel, RomReference],
        peak_tolerance_deg: float,
    ) -> None:
        self.settings = settings
        self.references = references
        self.peak_tolerance_deg = peak_tolerance_deg

    def segments(self, rep: CompletedRepetition) -> list[PhaseSegment]:
        """The three (or fewer) phase segments of the repetition."""
        samples = rep.samples
        last = len(samples) - 1
        peak_start, peak_end = rep.peak_start_idx, rep.peak_end_idx
        segments: list[PhaseSegment] = []
        if peak_start > 0:
            segments.append(
                PhaseSegment("ascent", 0, peak_start, samples[peak_start].angle_deg)
            )
        segments.append(
            PhaseSegment("peak_hold", peak_start, peak_end, samples[peak_end].angle_deg)
        )
        if peak_end < last:
            segments.append(
                PhaseSegment("descent", peak_end, last, samples[last].angle_deg)
            )
        return segments

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        samples = rep.samples
        last = len(samples) - 1
        peak_start, peak_end = rep.peak_start_idx, rep.peak_end_idx
        baseline = rep.baseline_angle

        ascent_rom = abs(samples[peak_start].angle_deg - baseline)
        descent_rom = abs(rep.peak_angle - samples[last].angle_deg)
        peak_hold_duration = samples[peak_end].timestamp - samples[peak_start].timestamp
        # ``peak_start_idx == 0`` means the very first recorded sample is
        # already inside the peak band: the ascent was not fully observed by
        # this rep segment. We still report the lower-bound ascent_rom
        # (samples[0] - baseline) honestly, but flag the structural truncation
        # so feedback can prompt the patient to start from neutral and stay
        # visible before moving.
        ascent_truncated = peak_start == 0
        detail: dict[str, float] = {
            "ascent_rom_deg": ascent_rom,
            "descent_rom_deg": descent_rom,
            "peak_hold_duration_s": peak_hold_duration,
            "ascent_truncated": 1.0 if ascent_truncated else 0.0,
        }

        reference = self.references.get(action)
        target = reference.target_deg if reference is not None else None
        if target is None or target <= 0:
            return MetricResult(
                name=self.name,
                status="reference_missing",
                passed=None,
                score=None,
                primary_value=min(ascent_rom, descent_rom),
                detail=detail,
            )

        phase_threshold = target * self.settings.phase_rom_ratio
        ascent_pass = ascent_rom >= phase_threshold
        descent_pass = descent_rom >= phase_threshold
        passed_count = int(ascent_pass) + int(descent_pass)
        detail["ascent_threshold_deg"] = phase_threshold
        detail["descent_threshold_deg"] = phase_threshold
        return MetricResult(
            name=self.name,
            status="ok",
            passed=ascent_pass and descent_pass,
            score=100.0 * passed_count / 2.0,
            primary_value=min(ascent_rom, descent_rom),
            detail=detail,
        )

    def debug_payload(
        self, rep: CompletedRepetition, action: ActionLabel
    ) -> PhaseRomDebugPayload:
        smoothed = [s.angle_deg for s in rep.samples]
        timestamps = [s.timestamp for s in rep.samples]
        reference = self.references.get(action)
        target = reference.target_deg if reference is not None else None
        threshold = (
            target * self.settings.phase_rom_ratio
            if (target is not None and target > 0)
            else None
        )
        return PhaseRomDebugPayload(
            smoothed_angles_deg=smoothed,
            timestamps_s=timestamps,
            baseline_deg=rep.baseline_angle,
            peak_deg=rep.peak_angle,
            peak_band_low_deg=rep.peak_angle - self.peak_tolerance_deg,
            peak_start_idx=rep.peak_start_idx,
            peak_end_idx=rep.peak_end_idx,
            ascent_threshold_deg=threshold,
            descent_threshold_deg=threshold,
        )

"""Turn failed metric results into reason codes and correction targets.

Most reason codes are gated on ``passed is False``: slots (``passed is None``)
and passing metrics are ignored. The exception is *structural* advisories such
as ``ascent_truncated``: evaluators set a flag in ``detail`` whenever the
measurement itself is structurally incomplete (e.g. the first recorded rep
sample is already in the peak band). These fire regardless of ``passed``
because they tell the patient how to perform the rep so the next measurement
is trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import MetricResult


@dataclass(slots=True)
class Feedback:
    reason_codes: list[str] = field(default_factory=list)
    correction_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_codes": list(self.reason_codes),
            "correction_targets": list(self.correction_targets),
        }


class FeedbackBuilder:
    def build(self, results: list[MetricResult]) -> Feedback:
        by_name = {result.name: result for result in results}
        reasons: list[str] = []
        targets: list[str] = []

        rom = by_name.get("rom")
        if rom is not None and rom.passed is False:
            reasons.append("rom_below_required")
            targets.append("increase_range_of_motion")

        phase = by_name.get("phase_rom")
        if phase is not None:
            detail = phase.detail
            if detail.get("ascent_truncated", 0.0) > 0.0:
                reasons.append("ascent_truncated")
                targets.append("start_from_neutral_and_keep_visible")
            if phase.passed is False:
                if detail.get("ascent_rom_deg", 0.0) < detail.get("ascent_threshold_deg", 0.0):
                    reasons.append("phase_rom_insufficient_ascent")
                    targets.append("improve_ascent_range")
                if detail.get("descent_rom_deg", 0.0) < detail.get("descent_threshold_deg", 0.0):
                    reasons.append("phase_rom_insufficient_descent")
                    targets.append("improve_descent_range")

        symmetry = by_name.get("symmetry")
        if symmetry is not None and symmetry.passed is False:
            reasons.append("symmetry_below_threshold")
            targets.append("balance_left_and_right_sides")

        compensation = by_name.get("compensation")
        if compensation is not None and compensation.passed is False:
            detail = compensation.detail
            if detail.get("trunk_tilt_detected", 0.0) > 0.0:
                reasons.append("trunk_lateral_shift_detected")
                targets.append("keep_trunk_upright")
            if detail.get("shoulder_elevation_detected", 0.0) > 0.0:
                reasons.append("shoulder_elevation_detected")
                targets.append("relax_shoulder")
        if (
            compensation is not None
            and compensation.detail.get("aux_shoulder_elevation_detected", 0.0) > 0.0
        ):
            reasons.append("contralateral_shoulder_elevation_observed")
            targets.append("relax_non_moving_shoulder")

        return Feedback(reason_codes=reasons, correction_targets=targets)

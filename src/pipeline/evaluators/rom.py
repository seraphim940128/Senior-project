"""ROM evaluator: |peak - baseline| versus the per-action required minimum."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from models import ActionLabel, CompletedRepetition, MetricResult, RomDebugPayload


@dataclass(slots=True)
class RomReference:
    required_min_deg: float | None
    target_deg: float | None
    source_title: str
    source_note: str


def load_rom_references(path: Path) -> dict[ActionLabel, RomReference]:
    """Load per-action ROM references from a YAML file."""
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    references: dict[ActionLabel, RomReference] = {}
    for action_name, raw in payload.items():
        references[ActionLabel.from_value(action_name)] = RomReference(
            required_min_deg=(
                float(raw["required_min_deg"])
                if raw.get("required_min_deg") is not None
                else None
            ),
            target_deg=float(raw["target_deg"]) if raw.get("target_deg") is not None else None,
            source_title=str(raw.get("source_title", "")),
            source_note=str(raw.get("source_note", "")),
        )
    return references


# pass/fail score mapping.
_PASS_SCORE = 100.0
_FAIL_SCORE = 40.0


class RomEvaluator:
    name = "rom"

    def __init__(self, references: dict[ActionLabel, RomReference]) -> None:
        self.references = references

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        rom = rep.dynamic_rom  # |peak - baseline|
        detail: dict[str, float] = {
            "baseline_deg": rep.baseline_angle,
            "peak_deg": rep.peak_angle,
        }
        reference = self.references.get(action)
        if reference is None or reference.required_min_deg is None:
            return MetricResult(
                name=self.name,
                status="reference_missing",
                passed=None,
                score=None,
                primary_value=rom,
                detail=detail,
            )
        passed = rom >= reference.required_min_deg
        detail["required_min_deg"] = reference.required_min_deg
        if reference.target_deg is not None:
            detail["target_deg"] = reference.target_deg
        return MetricResult(
            name=self.name,
            status="ok",
            passed=passed,
            score=_PASS_SCORE if passed else _FAIL_SCORE,
            primary_value=rom,
            detail=detail,
        )

    def debug_payload(self, rep: CompletedRepetition, action: ActionLabel) -> RomDebugPayload:
        smoothed = [s.angle_deg for s in rep.samples]
        raw = [
            s.raw_angle_deg if s.raw_angle_deg is not None else s.angle_deg
            for s in rep.samples
        ]
        timestamps = [s.timestamp for s in rep.samples]
        peak_idx = max(range(len(smoothed)), key=lambda i: smoothed[i]) if smoothed else 0
        reference = self.references.get(action)
        return RomDebugPayload(
            raw_angles_deg=raw,
            smoothed_angles_deg=smoothed,
            timestamps_s=timestamps,
            baseline_deg=rep.baseline_angle,
            peak_idx=peak_idx,
            threshold_min_deg=reference.required_min_deg if reference else None,
        )

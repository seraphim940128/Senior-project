"""Weighted aggregation of metric scores into one overall 0-100 score.

Statuses are honored explicitly so the overall score does not silently reward
the patient for skipping calibration or for an evaluator that could not score:

* ``_SKIP_STATUSES`` (``not_applicable`` / ``reference_missing`` /
  ``not_implemented``) drop out of the weighted denominator so the remaining
  metrics renormalize. These statuses mean the metric did not apply this
  time (bilateral metric on a single-arm action, missing YAML reference,
  unimplemented slot) - they are not the patient's fault.

* Any other status (most importantly ``unreliable``) contributes ``0`` with
  the full configured weight. A patient who skips calibration must not get
  the same overall score as one who calibrated and scored well on
  compensation.

Only ``status == "ok"`` with a finite numeric score is admitted as a real
score; everything else is treated defensively as 0 within an active weight,
unless explicitly listed in ``_SKIP_STATUSES``.
"""
from __future__ import annotations

import math

from models import MetricResult


class ScoreAggregator:
    # Weight per metric. dtw / deviation are omitted until their slots are
    # implemented; add them here at that point.
    _WEIGHTS: dict[str, float] = {
        "rom": 2.0,
        "phase_rom": 1.0,
        "symmetry": 0.5,
        "compensation": 1.0,
    }
    # Statuses that mean "this metric does not apply this time": drop from the
    # denominator so the remaining metrics renormalize.
    _SKIP_STATUSES: frozenset[str] = frozenset(
        ("not_applicable", "reference_missing", "not_implemented")
    )

    def aggregate(self, results: list[MetricResult]) -> float | None:
        """Weighted mean of the scored metrics, or ``None`` if no metric
        produced a real ``ok`` score (all skipped or all unreliable).
        """
        weighted: list[tuple[float, float]] = []
        has_any_real_score = False
        for result in results:
            if result.name not in self._WEIGHTS:
                continue
            weight = self._WEIGHTS[result.name]
            if result.status in self._SKIP_STATUSES:
                continue
            if (
                result.status == "ok"
                and result.score is not None
                and math.isfinite(result.score)
            ):
                weighted.append((result.score, weight))
                has_any_real_score = True
            else:
                weighted.append((0.0, weight))
        if not weighted or not has_any_real_score:
            return None
        total_weight = sum(weight for _, weight in weighted)
        raw = sum(score * weight for score, weight in weighted) / total_weight
        return max(0.0, min(100.0, raw))

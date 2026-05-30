"""Metric evaluator interface.

Every evaluator takes one ``CompletedRepetition`` and returns:
  - ``evaluate``      -> a unified ``MetricResult`` (scalar score / pass / value)
  - ``debug_payload`` -> a per-metric ``DebugPayload`` (viz / diagnostic series)

The two outputs are independent. ``DebugPayload`` carries time-series for live
charting and MUST NOT reach the scoring layer (enforced by
``tests/test_no_debug_in_score.py``).
"""
from __future__ import annotations

from typing import Protocol

from models import ActionLabel, CompletedRepetition, DebugPayload, MetricResult


class MetricEvaluator(Protocol):
    """Structural interface implemented by every evaluator and every slot."""

    name: str

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        ...

    def debug_payload(self, rep: CompletedRepetition, action: ActionLabel) -> DebugPayload:
        ...


class NotImplementedEvaluator:
    """Placeholder slot — structurally a ``MetricEvaluator`` that does nothing.

    The score aggregator recognizes ``status="not_implemented"`` and skips it,
    so a slot can be wired into the pipeline before its real logic exists.
    Concrete slots subclass this and set their own ``name``.
    """

    name = "slot"

    def evaluate(self, rep: CompletedRepetition, action: ActionLabel) -> MetricResult:
        return MetricResult(
            name=self.name,
            status="not_implemented",
            passed=None,
            score=None,
            primary_value=None,
            detail={},
        )

    def debug_payload(self, rep: CompletedRepetition, action: ActionLabel) -> DebugPayload:
        return DebugPayload()

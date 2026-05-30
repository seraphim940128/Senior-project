"""Deviation evaluator — architectural slot.

Logic intentionally not implemented. See ``base.py``
(``NotImplementedEvaluator``) for the slot pattern.
"""
from __future__ import annotations

from pipeline.evaluators.base import NotImplementedEvaluator


class DeviationEvaluator(NotImplementedEvaluator):
    name = "deviation"

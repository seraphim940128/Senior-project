"""DTW evaluator — architectural slot.

Logic intentionally not implemented. See ``base.py``
(``NotImplementedEvaluator``) for the slot pattern.
"""
from __future__ import annotations

from pipeline.evaluators.base import NotImplementedEvaluator


class DtwEvaluator(NotImplementedEvaluator):
    name = "dtw"

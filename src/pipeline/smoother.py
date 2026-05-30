"""Dual-track signal smoothing.

- physics track: sliding-median filter. Feeds the state machine and every
  evaluator. Edge-preserving, no amplitude attenuation.
- visualization track: EMA. Patient-facing rendering ONLY — it flattens peaks
  and must never reach an evaluator or the state machine.

Each track has a batch function (for evaluating a finished repetition) and an
online class (for the per-frame live pipeline).
"""
from __future__ import annotations

from collections import deque
from statistics import median


# ---------------------------------------------------------------------------
# batch — physics track
# ---------------------------------------------------------------------------

def median_filter(values: list[float], window: int) -> list[float]:
    """Centered sliding-window median. Near the edges the window shrinks
    (no padding distortion).
    """
    if not values or window <= 1:
        return [float(v) for v in values]
    n = len(values)
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(float(median(values[lo:hi])))
    return out


def physics_smooth(values: list[float], median_window: int = 5) -> list[float]:
    """Physics-track smoothing — the only smoother evaluators may use."""
    return median_filter(values, median_window)


# ---------------------------------------------------------------------------
# batch — visualization track
# ---------------------------------------------------------------------------

def ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average. VISUALIZATION ONLY — never for evaluation."""
    if not values or alpha >= 1.0:
        return [float(v) for v in values]
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


# ---------------------------------------------------------------------------
# online — per-frame
# ---------------------------------------------------------------------------

class OnlineMedianSmoother:
    """Per-frame sliding-median smoother for the physics track."""

    def __init__(self, window: int) -> None:
        self._window = max(1, int(window))
        self._buffer: deque[float] = deque(maxlen=self._window)

    def push(self, value: float) -> float:
        self._buffer.append(float(value))
        return float(median(self._buffer))

    def reset(self) -> None:
        self._buffer.clear()


class OnlineEmaSmoother:
    """Per-frame EMA smoother for the visualization track ONLY."""

    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self._value: float | None = None

    def push(self, value: float) -> float:
        if self._value is None:
            self._value = float(value)
        else:
            self._value = self._alpha * float(value) + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None

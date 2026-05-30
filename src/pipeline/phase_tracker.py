"""Online phase / repetition state machine — the single source of truth.

Both the production evaluation path and the live overlay consume this one
tracker; there is exactly one phase state machine in the project (enforced by
``tests/test_single_state_machine.py``).

State graph::

    idle --> ascent --> peak --> descent --> idle (emit CompletedRepetition)

Because :mod:`angle_extractor` unifies every action onto an *increasing*
primary angle (elbow flexion is inverted), the tracker needs no per-action
direction handling: ascent always means the primary angle rises, neutral
always means it is low — so there is no signed-progress machinery or
per-action neutral gate.

The cooldown / max-baseline / min-dynamic-ROM onset guards gate when a new
repetition is allowed to start.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from config import PhaseTrackerSettings
from models import ActionLabel, CompletedRepetition, Landmark, PoseSample


@dataclass(frozen=True, slots=True)
class PhaseSnapshot:
    """Per-frame view of tracker state, consumed by the live overlay.

    ``dropped_due_to_missing`` is set on the *single* frame where the
    in-flight cycle was abandoned because ``missing_counter`` exceeded
    ``missing_grace_frames``. It exists so the overlay / telemetry can
    surface the event ("rep discarded - lost tracking") instead of the
    cycle silently disappearing.
    """

    current_phase: str               # "idle" | "ascent" | "peak" | "descent"
    waiting_for_neutral: bool
    neutral_ready: bool
    current_angle: float | None
    current_delta: float
    live_rom: float | None           # running peak-minus-baseline of the in-flight rep
    ascent_frames: int
    peak_frames: int
    descent_frames: int
    missing_counter: int
    dropped_due_to_missing: bool = False


class PhaseTracker:
    """Online state machine over the physics-track (median-smoothed) angle."""

    def __init__(self, settings: PhaseTrackerSettings, action: ActionLabel) -> None:
        self.settings = settings
        self.action = action
        self.reset()

    # -- public API -----------------------------------------------------------

    def reset(self) -> None:
        self.current_phase = "idle"
        self.waiting_for_neutral = True
        self.neutral_ready = False
        self.current_angle: float | None = None
        self.current_delta = 0.0
        self._prev_angle: float | None = None
        self._neutral_counter = 0
        self._neutral_values: list[float] = []
        self._armed_baseline: float | None = None
        self._last_ended_at: float | None = None
        self._completed: CompletedRepetition | None = None
        # One-shot telemetry flag - True only on the single frame where the
        # in-flight cycle was abandoned because of missing-frame timeout.
        self._dropped_due_to_missing = False
        # Schmitt-trigger state for "in neutral band": start outside so the
        # very first frame must clear the strict ``lower`` (= max - hysteresis)
        # to enter; once inside, stay inside until we cross ``upper`` (= max).
        # See _update_neutral_arming.
        self._in_neutral_band = False
        self._reset_cycle()

    @property
    def completed(self) -> CompletedRepetition | None:
        """The repetition emitted on this frame, or ``None``. Valid until the
        next :meth:`update` call.
        """
        return self._completed

    def update(
        self,
        timestamp: float,
        smoothed_angle: float | None,
        landmarks: dict[str, Landmark] | None,
        image_landmarks: dict[str, Landmark] | None = None,
        raw_angle: float | None = None,
    ) -> PhaseSnapshot:
        """Advance the state machine by one frame.

        ``smoothed_angle`` is the physics-track (median-smoothed) primary angle;
        ``None`` (or ``None`` landmarks) marks a dropped frame. ``raw_angle`` is
        the pre-smoothing value, stored on the sample for debug charts only.
        """
        self._completed = None
        self._dropped_due_to_missing = False

        if smoothed_angle is None or landmarks is None:
            return self._handle_missing()

        # Amortize the post-gap delta over the gap length: a raw
        # ``smoothed - prev`` after N dropped frames represents motion over
        # N+1 frame intervals, not 1. Dividing by ``missing_counter + 1``
        # prevents the next-frame delta from looking like a single huge
        # jump, which would otherwise spuriously trigger the ascent->peak
        # hysteresis ("peak_pending") or the peak->descent gate
        # (``current_delta < 0``). The state machine should not change
        # phase on a gap artefact.
        if self._prev_angle is None:
            delta = 0.0
        else:
            delta = (smoothed_angle - self._prev_angle) / (self._missing_counter + 1)
        self._missing_counter = 0
        self.current_angle = float(smoothed_angle)
        self.current_delta = float(delta)

        self._update_neutral_arming(self.current_angle)

        sample = PoseSample(
            timestamp=float(timestamp),
            angle_deg=self.current_angle,
            delta_deg=self.current_delta,
            landmarks=dict(landmarks),
            image_landmarks=dict(image_landmarks) if image_landmarks else None,
            raw_angle_deg=float(raw_angle) if raw_angle is not None else self.current_angle,
        )
        # Accumulate into an in-flight cycle before dispatch so the completing
        # frame is already in ``_samples`` when ``_complete`` runs.
        if self.current_phase != "idle":
            self._samples.append(sample)

        if self.current_phase == "idle":
            self._handle_idle(timestamp, sample)
        elif self.current_phase == "ascent":
            self._handle_ascent()
        elif self.current_phase == "peak":
            self._handle_peak()
        elif self.current_phase == "descent":
            self._handle_descent(timestamp)

        self._prev_angle = self.current_angle
        return self.snapshot()

    def snapshot(self) -> PhaseSnapshot:
        live_rom: float | None = None
        if self._cycle_baseline is not None and self.current_phase != "idle":
            live_rom = max(0.0, self._running_max_angle - self._cycle_baseline)
        return PhaseSnapshot(
            current_phase=self.current_phase,
            waiting_for_neutral=self.waiting_for_neutral,
            neutral_ready=self.neutral_ready,
            current_angle=self.current_angle,
            current_delta=self.current_delta,
            live_rom=live_rom,
            ascent_frames=self._ascent_frames,
            peak_frames=self._peak_frames,
            descent_frames=self._descent_frames,
            missing_counter=self._missing_counter,
            dropped_due_to_missing=self._dropped_due_to_missing,
        )

    # -- internals ------------------------------------------------------------

    def _reset_cycle(self) -> None:
        self._samples: list[PoseSample] = []
        self._cycle_baseline: float | None = None
        self._running_max_angle = 0.0
        self._ascent_frames = 0
        self._peak_frames = 0
        self._descent_frames = 0
        self._peak_pending = 0
        self._missing_counter = 0

    def _update_neutral_arming(self, angle: float) -> None:
        """Arm ``_armed_baseline`` once the limb has held a neutral pose.

        "In neutral" is decided by a Schmitt trigger so a frame oscillating
        around ``neutral_max_deg`` does not chatter the neutral confirm
        counter on/off (which would otherwise re-arm a fake baseline
        mid-rep). Two thresholds:

        - ``upper = neutral_max_deg`` - while currently in-neutral, must
          rise *above* this to leave.
        - ``lower = neutral_max_deg - hysteresis_band_deg`` - while
          currently out-of-neutral, must drop *to/below* this to re-enter.

        After re-entry, ``neutral_confirm_frames`` consecutive in-neutral
        frames are still required before ``neutral_ready`` flips and the
        new baseline is armed from their mean.
        """
        upper = self.settings.neutral_max_deg
        lower = upper - self.settings.hysteresis_band_deg
        if self._in_neutral_band:
            self._in_neutral_band = angle <= upper
        else:
            self._in_neutral_band = angle <= lower
        if self._in_neutral_band:
            self._neutral_counter += 1
            self._neutral_values.append(angle)
            if len(self._neutral_values) > self.settings.neutral_confirm_frames:
                self._neutral_values.pop(0)
        else:
            self._neutral_counter = 0
            self._neutral_values.clear()
        self.neutral_ready = self._neutral_counter >= self.settings.neutral_confirm_frames
        if self.neutral_ready and self._neutral_values:
            self._armed_baseline = float(mean(self._neutral_values))

    def _handle_missing(self) -> PhaseSnapshot:
        self.current_angle = None
        self.current_delta = 0.0
        if self.current_phase != "idle":
            self._missing_counter += 1
            if self._missing_counter > self.settings.missing_grace_frames:
                # Beyond grace: drop the in-flight cycle, emit nothing,
                # surface ONE telemetry flag so the overlay/log can show
                # "rep discarded - lost tracking" instead of the cycle
                # silently disappearing.
                self._reset_cycle()
                self.current_phase = "idle"
                self._dropped_due_to_missing = True
        self.waiting_for_neutral = (
            self.current_phase == "idle" and self._armed_baseline is None
        )
        return self.snapshot()

    def _handle_idle(self, timestamp: float, sample: PoseSample) -> None:
        self.waiting_for_neutral = self._armed_baseline is None
        if self._armed_baseline is None or self.current_angle is None:
            return
        progress = self.current_angle - self._armed_baseline
        if progress <= self.settings.neutral_exit_delta_deg or self.current_delta <= 0:
            return
        # rep-onset guards
        cooldown_ok = (
            self._last_ended_at is None
            or (timestamp - self._last_ended_at) >= self.settings.min_cooldown_s
        )
        baseline_ok = self._armed_baseline <= self.settings.max_baseline_deg
        if not (cooldown_ok and baseline_ok):
            return
        # idle -> ascent
        self._reset_cycle()
        self.current_phase = "ascent"
        self.waiting_for_neutral = False
        self._cycle_baseline = self._armed_baseline
        self._running_max_angle = self.current_angle
        self._ascent_frames = 1
        self._samples = [sample]

    def _handle_ascent(self) -> None:
        self._ascent_frames += 1
        if self.current_angle is not None and self.current_angle > self._running_max_angle:
            self._running_max_angle = self.current_angle
        # Hysteresis: a sustained reversal (not a single noisy dip) commits to
        # peak. A fresh advance clears the counter; a pause holds it.
        if self.current_delta > self.settings.pause_epsilon_deg:
            self._peak_pending = 0
        elif self.current_delta < -self.settings.pause_epsilon_deg:
            self._peak_pending += 1
        if (
            self._ascent_frames >= self.settings.min_ascent_frames
            and self._peak_pending >= self.settings.peak_confirm_frames
        ):
            self.current_phase = "peak"
            self._peak_frames = 1

    def _handle_peak(self) -> None:
        self._peak_frames += 1
        if self.current_angle is not None and self.current_angle > self._running_max_angle:
            self._running_max_angle = self.current_angle
        drop = self._running_max_angle - (self.current_angle or self._running_max_angle)
        if (
            self._peak_frames >= self.settings.min_peak_frames
            and drop > self.settings.peak_tolerance_deg
            and self.current_delta < 0
        ):
            self.current_phase = "descent"
            self._descent_frames = 1
            self._peak_pending = 0

    def _handle_descent(self, timestamp: float) -> None:
        self._descent_frames += 1
        if (
            self._descent_frames >= self.settings.min_descent_frames
            and self.neutral_ready
        ):
            self._complete(timestamp)

    def _complete(self, timestamp: float) -> None:
        """Finish the in-flight cycle: emit a CompletedRepetition unless it
        fails the min-frames / min-dynamic-ROM guards.
        """
        samples = self._samples
        baseline = self._cycle_baseline
        if baseline is not None and samples:
            peak_angle = max(s.angle_deg for s in samples)
            dynamic_rom = peak_angle - baseline
            if (
                dynamic_rom >= self.settings.min_dynamic_rom_deg
                and len(samples) >= self.settings.min_frames
            ):
                threshold = peak_angle - self.settings.peak_tolerance_deg
                in_band = [i for i, s in enumerate(samples) if s.angle_deg >= threshold]
                self._completed = CompletedRepetition(
                    action=self.action,
                    samples=list(samples),
                    baseline_angle=float(baseline),
                    peak_angle=float(peak_angle),
                    movement_direction=1,
                    peak_start_idx=in_band[0],
                    peak_end_idx=in_band[-1],
                    started_at=samples[0].timestamp,
                    ended_at=samples[-1].timestamp,
                )
        self._last_ended_at = float(timestamp)
        self.current_phase = "idle"
        self.waiting_for_neutral = False
        self._reset_cycle()

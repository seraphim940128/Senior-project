"""LiveSession — the per-frame orchestrator.

Wires the per-frame pipeline for one pre-selected action::

    pose_detector -> quality_filter -> angle_extractor -> smoother -> PhaseTracker

On every completed repetition it runs all metric evaluators, aggregates a score
and derives feedback. Pure logic: no rendering, no camera I/O (those live in
``overlay_renderer`` / ``app_runner``), so this whole class is unit-testable.

Compensation detection needs a session-start resting-posture baseline. The
session therefore has a small *calibration mode* (a plain state field, not a
second state machine — the single rep-segmentation state machine remains
``PhaseTracker``)::

    awaiting -> countdown -> collecting -> active

``calibration="manual"`` starts in ``awaiting`` (the runner calls
``start_calibration`` on the R key); ``"auto"`` skips straight to ``collecting``
(headless video); ``"off"`` (the default, used by the unit tests) runs the
pipeline immediately with no baseline — compensation then reports ``unreliable``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import AppSettings
from models import (
    ActionLabel,
    CalibrationStatus,
    DebugPayload,
    JointReadout,
    Landmark,
    LiveCompensation,
    MetricResult,
)
from pipeline.angle_extractor import (
    bilateral_shoulder_angles,
    measured_joints,
    primary_angle,
)
from pipeline.compensation_baseline import BaselineCollector, PostureBaseline
from pipeline.evaluators.base import MetricEvaluator
from pipeline.evaluators.compensation import (
    CompensationEvaluator,
    compensation_aux_shoulder_side,
    compensation_checks_trunk,
    compensation_shoulder_sides,
    compensation_uses_ascent_phase,
)
from pipeline.evaluators.deviation import DeviationEvaluator
from pipeline.evaluators.dtw import DtwEvaluator
from pipeline.evaluators.phase_rom import PhaseRomEvaluator
from pipeline.evaluators.rom import RomEvaluator, load_rom_references
from pipeline.evaluators.symmetry import SymmetryEvaluator
from pipeline.phase_tracker import PhaseSnapshot, PhaseTracker
from pipeline.pose_detector import PoseTracker
from pipeline.posture_metrics import shoulder_hip_offset, trunk_lateral_tilt_deg
from pipeline.quality_filter import available_landmarks, required_action_landmarks_present
from pipeline.smoother import OnlineEmaSmoother, OnlineMedianSmoother
from scoring.aggregator import ScoreAggregator
from scoring.feedback import Feedback, FeedbackBuilder

# calibration mode -> initial calibration state
_CALIBRATING_STATES = ("awaiting", "countdown", "collecting")


@dataclass(slots=True)
class SessionSnapshot:
    """Result of processing one frame."""

    phase: PhaseSnapshot
    raw_angle: float | None
    viz_angle: float | None                       # EMA-smoothed, display only
    rep_results: list[MetricResult] | None        # set only on a completed-rep frame
    rep_debug: dict[str, DebugPayload] | None
    overall_score: float | None
    feedback: Feedback | None
    pose_xy: dict[str, tuple[float, float]] | None = None      # landmark pixels
    joint_readouts: list[JointReadout] | None = None           # measured-joint angles
    calibration: CalibrationStatus | None = None               # None when mode is off
    live_compensation: LiveCompensation | None = None           # per-frame readout


def build_evaluators(settings: AppSettings) -> list[MetricEvaluator]:
    """The full evaluator set: four core metrics plus two placeholder slots.

    ROM / phase-ROM use ``settings.pose.basis`` (kinematics, 2D). Symmetry uses
    ``settings.pose.symmetry_basis`` (3D) and compensation uses
    ``settings.pose.compensation_basis`` (3D), so each left/right or geometric
    proxy is measured where it is most stable, independent of the kinematics basis.
    """
    references = load_rom_references(settings.paths.rom_reference_path)
    return [
        RomEvaluator(references),
        PhaseRomEvaluator(
            settings.phase_rom, references, settings.phase_tracker.peak_tolerance_deg
        ),
        SymmetryEvaluator(
            settings.symmetry, settings.pose.symmetry_basis, settings.signal.median_window
        ),
        CompensationEvaluator(
            settings.compensation,
            settings.pose.compensation_basis,
            settings.signal.median_window,
        ),
        DeviationEvaluator(),
        DtwEvaluator(),
    ]


class LiveSession:
    def __init__(
        self,
        settings: AppSettings,
        pose_tracker: PoseTracker,
        action: ActionLabel,
        calibration: str = "off",
    ) -> None:
        if calibration not in {"off", "manual", "auto"}:
            raise ValueError(f"calibration must be off/manual/auto, got {calibration!r}")
        self.settings = settings
        self.pose_tracker = pose_tracker
        self.basis = settings.pose.basis              # kinematics: ROM/segmentation
        self.comp_basis = settings.pose.compensation_basis  # compensation + calibration
        self.sym_basis = settings.pose.symmetry_basis       # symmetry evaluator (left/right comparison)
        self.action = action
        self._calibration_mode = calibration
        self._aggregator = ScoreAggregator()
        self._feedback = FeedbackBuilder()
        self._evaluators = build_evaluators(settings)
        self._compensation: CompensationEvaluator | None = next(
            (e for e in self._evaluators if isinstance(e, CompensationEvaluator)), None
        )
        self._symmetry: SymmetryEvaluator | None = next(
            (e for e in self._evaluators if isinstance(e, SymmetryEvaluator)), None
        )
        self._build_runtime()
        self._init_calibration()

    # -- lifecycle ------------------------------------------------------------

    def _build_runtime(self) -> None:
        self._median = OnlineMedianSmoother(self.settings.signal.median_window)
        self._ema = OnlineEmaSmoother(self.settings.signal.viz_ema_alpha)
        self._tracker = PhaseTracker(self.settings.phase_tracker, self.action)

    def _new_collector(self) -> BaselineCollector:
        baseline = self.settings.compensation.baseline
        return BaselineCollector(
            baseline.target_frames,
            self.comp_basis,
            baseline.max_trunk_tilt_spread_deg,
            baseline.max_shoulder_offset_spread_ratio,
        )

    def _init_calibration(self) -> None:
        """Set the calibration state for the current mode and clear the baseline."""
        self._collector: BaselineCollector | None = None
        self._countdown_start: float | None = None
        self._baseline: PostureBaseline | None = None
        self._awaiting_message = "Stand ready, then press R"
        self.baseline_unsteady = False
        if self._compensation is not None:
            self._compensation.baseline = None
        if self._calibration_mode == "manual":
            self._calib_state = "awaiting"
        elif self._calibration_mode == "auto":
            self._calib_state = "collecting"
            self._collector = self._new_collector()
        else:
            self._calib_state = "off"

    def set_action(self, action: ActionLabel) -> None:
        self.action = action
        self._build_runtime()
        self._init_calibration()

    def reset(self) -> None:
        self._build_runtime()
        self._init_calibration()

    def start_calibration(self, skip_countdown: bool = False) -> None:
        """Begin (or restart) the resting-posture calibration. Triggered by the
        R key in manual mode; ``skip_countdown`` is used by headless video.
        """
        self._build_runtime()
        self._collector = self._new_collector()
        self._countdown_start = None
        self._baseline = None
        self.baseline_unsteady = False
        if self._compensation is not None:
            self._compensation.baseline = None
        self._calib_state = "collecting" if skip_countdown else "countdown"

    @property
    def is_calibrating(self) -> bool:
        return self._calib_state in _CALIBRATING_STATES

    @property
    def baseline(self) -> PostureBaseline | None:
        return self._baseline

    # -- per-frame ------------------------------------------------------------

    def process_frame(self, frame: object, timestamp: float) -> SessionSnapshot:
        pose_frame = self.pose_tracker.extract(frame)
        lm3d: dict[str, Landmark] | None = None
        lm2d: dict[str, Landmark] | None = None
        if pose_frame is not None:
            min_vis = self.settings.pose.min_landmark_visibility
            min_pres = self.settings.pose.min_landmark_presence
            lm3d = available_landmarks(pose_frame, min_vis, min_pres, "3d")
            lm2d = available_landmarks(pose_frame, min_vis, min_pres, "2d")

        if self._calib_state in _CALIBRATING_STATES:
            return self._calibration_step(timestamp, lm2d, lm3d)

        basis_lm = lm2d if self.basis == "2d" else lm3d
        if not basis_lm or not required_action_landmarks_present(self.action, basis_lm):
            return self._on_missing(timestamp)

        raw = primary_angle(self.action, basis_lm, self.basis)
        if raw is None:
            return self._on_missing(timestamp)

        smoothed = self._median.push(raw)
        viz = self._ema.push(raw)
        phase = self._tracker.update(timestamp, smoothed, lm3d, lm2d, raw_angle=raw)
        return self._finish(phase, raw, viz, lm2d, lm3d)

    # -- calibration ----------------------------------------------------------

    def _calibration_step(
        self,
        timestamp: float,
        lm2d: dict[str, Landmark] | None,
        lm3d: dict[str, Landmark] | None,
    ) -> SessionSnapshot:
        """Advance the resting-posture calibration by one frame."""
        calib_lm = lm2d if self.comp_basis == "2d" else lm3d
        countdown_seconds = self.settings.compensation.baseline.countdown_seconds

        if self._calib_state == "countdown":
            if self._countdown_start is None:
                self._countdown_start = timestamp
            if timestamp - self._countdown_start >= countdown_seconds:
                self._calib_state = "collecting"

        if self._calib_state == "collecting" and calib_lm:
            assert self._collector is not None
            self._collector.push(calib_lm)
            if self._collector.ready:
                self._complete_or_retry_calibration()

        return SessionSnapshot(
            phase=self._tracker.snapshot(),
            raw_angle=None,
            viz_angle=None,
            rep_results=None,
            rep_debug=None,
            overall_score=None,
            feedback=None,
            pose_xy=self._pose_xy(lm2d),
            joint_readouts=None,
            calibration=self._calibration_status(timestamp),
            live_compensation=None,
        )

    def _complete_or_retry_calibration(self) -> None:
        """Enough frames collected — accept the baseline if the resting frames
        were steady, otherwise handle the unsteady case per calibration mode.
        """
        assert self._collector is not None
        if self._collector.steady_ok:
            self._finish_calibration()
            return
        if self._calibration_mode == "auto":
            # Headless video: no user to re-prompt — accept but flag low
            # confidence so the runner can warn (see app.py).
            self.baseline_unsteady = True
            self._finish_calibration()
        else:
            # Manual: discard and ask the user to redo the calibration.
            self._collector = None
            self._countdown_start = None
            self._calib_state = "awaiting"
            self._awaiting_message = (
                "Hold still — calibration unsteady. Press R to retry"
            )

    def _finish_calibration(self) -> None:
        assert self._collector is not None
        self._baseline = self._collector.build()
        if self._compensation is not None:
            self._compensation.baseline = self._baseline
        self._calib_state = "active"

    def _calibration_status(self, timestamp: float) -> CalibrationStatus | None:
        state = self._calib_state
        if state == "off":
            return None
        if state == "awaiting":
            return CalibrationStatus(state, self._awaiting_message)
        if state == "countdown":
            countdown_seconds = self.settings.compensation.baseline.countdown_seconds
            start = self._countdown_start if self._countdown_start is not None else timestamp
            remaining = max(0.0, countdown_seconds - (timestamp - start))
            return CalibrationStatus(
                state,
                f"Get ready: {max(1, math.ceil(remaining))}",
                countdown_remaining_s=remaining,
            )
        if state == "collecting":
            assert self._collector is not None
            return CalibrationStatus(
                state,
                f"Calibrating {self._collector.count}/{self._collector.target}",
                frames_collected=self._collector.count,
                frames_target=self._collector.target,
            )
        return CalibrationStatus("active", "")

    # -- completion helpers ---------------------------------------------------

    def _on_missing(self, timestamp: float) -> SessionSnapshot:
        phase = self._tracker.update(timestamp, None, None)
        return SessionSnapshot(
            phase=phase, raw_angle=None, viz_angle=None,
            rep_results=None, rep_debug=None, overall_score=None, feedback=None,
            calibration=self._calibration_status(timestamp),
        )

    def _finish(
        self,
        phase: PhaseSnapshot,
        raw: float,
        viz: float,
        lm2d: dict[str, Landmark] | None,
        lm3d: dict[str, Landmark] | None,
    ) -> SessionSnapshot:
        pose_xy = self._pose_xy(lm2d)
        joint_readouts = self._build_joint_readouts(lm2d, lm3d)
        comp_lm = lm2d if self.comp_basis == "2d" else lm3d
        live_comp = self._live_compensation(comp_lm, phase)
        calibration = self._calibration_status(0.0)
        completed = self._tracker.completed
        if completed is None:
            return SessionSnapshot(
                phase=phase, raw_angle=raw, viz_angle=viz,
                rep_results=None, rep_debug=None, overall_score=None, feedback=None,
                pose_xy=pose_xy, joint_readouts=joint_readouts,
                calibration=calibration, live_compensation=live_comp,
            )
        # Symmetry now reads left/right onsets directly from rep.samples
        # (primary_angle = max(L,R) puts both arms' rises inside the rep window),
        # so no pre-rep buffer injection is needed here. See symmetry.py docstring.
        results = [ev.evaluate(completed, self.action) for ev in self._evaluators]
        debug = {
            ev.name: ev.debug_payload(completed, self.action)
            for ev in self._evaluators
        }
        return SessionSnapshot(
            phase=phase, raw_angle=raw, viz_angle=viz,
            rep_results=results, rep_debug=debug,
            overall_score=self._aggregator.aggregate(results),
            feedback=self._feedback.build(results),
            pose_xy=pose_xy, joint_readouts=joint_readouts,
            calibration=calibration, live_compensation=live_comp,
        )

    def _live_compensation(
        self, basis_lm: dict[str, Landmark] | None, phase: PhaseSnapshot
    ) -> LiveCompensation | None:
        """Cheap per-frame compensation readout against the calibrated baseline.

        Stateless: it reuses the same ``posture_metrics`` functions the per-rep
        ``CompensationEvaluator`` scores on, so the live warning cannot diverge
        from the scored result. ``None`` until a baseline exists.

        For shoulder actions, the ``shoulder_elevation_over_threshold`` warning
        is gated to the **setting phase** — the early ascent while the live ROM
        is still within ``shoulder_elevation_setting_phase_max_deg`` of rest.
        This mirrors the per-rep evaluator's setting-phase SE restriction so the
        live red flag does not accuse the normal scapular rhythm that lifts the
        shoulder landmark at higher elevation.
        """
        baseline = self._baseline
        if baseline is None or not basis_lm:
            return None
        comp = self.settings.compensation
        # Apply the SAME per-action rule the per-rep evaluator scores on, so the
        # live warning never disagrees with the rep result: trunk only when the
        # action checks it, shoulder only on the patient side(s).
        trunk_dev: float | None = None
        if compensation_checks_trunk(self.action):
            tilt = trunk_lateral_tilt_deg(basis_lm)
            trunk_dev = None if tilt is None else abs(tilt - baseline.trunk_tilt_deg)
        ratios: list[float] = []
        for side in compensation_shoulder_sides(self.action):
            offset = shoulder_hip_offset(basis_lm, side)
            if offset is not None:
                base_offset = baseline.shoulder_hip_offset_for(side)
                up_dir = 1.0 if base_offset >= 0.0 else -1.0
                rise = (offset - base_offset) * up_dir
                ratios.append(rise / baseline.shoulder_width)
        shoulder_ratio = max(ratios) if ratios else None
        aux_ratio: float | None = None
        aux_side = compensation_aux_shoulder_side(self.action)
        if aux_side is not None:
            offset = shoulder_hip_offset(basis_lm, aux_side)
            if offset is not None:
                base_offset = baseline.shoulder_hip_offset_for(aux_side)
                up_dir = 1.0 if base_offset >= 0.0 else -1.0
                aux_ratio = (offset - base_offset) * up_dir / baseline.shoulder_width
        shoulder_thr = comp.shoulder_elevation_threshold_ratio
        in_detection_window = True
        if compensation_uses_ascent_phase(self.action):
            in_detection_window = (
                phase.current_phase == "ascent"
                and phase.live_rom is not None
                and phase.live_rom <= comp.shoulder_elevation_setting_phase_max_deg
            )
        return LiveCompensation(
            trunk_tilt_deviation_deg=trunk_dev,
            shoulder_elevation_ratio=shoulder_ratio,
            aux_shoulder_elevation_ratio=aux_ratio,
            trunk_tilt_over_threshold=(
                trunk_dev is not None and trunk_dev > comp.trunk_tilt_threshold_deg
            ),
            shoulder_elevation_over_threshold=(
                shoulder_ratio is not None
                and shoulder_ratio > shoulder_thr
                and in_detection_window
            ),
            aux_shoulder_elevation_over_threshold=(
                aux_ratio is not None
                and aux_ratio > shoulder_thr
                and in_detection_window
            ),
        )

    @staticmethod
    def _pose_xy(
        lm2d: dict[str, Landmark] | None,
    ) -> dict[str, tuple[float, float]] | None:
        """Pixel (x, y) of every detected landmark — for the skeleton overlay."""
        if not lm2d:
            return None
        return {name: (float(c[0]), float(c[1])) for name, c in lm2d.items()}

    def _build_joint_readouts(
        self,
        lm2d: dict[str, Landmark] | None,
        lm3d: dict[str, Landmark] | None,
    ) -> list[JointReadout] | None:
        """Measured-joint angles for the live overlay.

        Reuses the exact pipeline functions the evaluators / state machine use
        (``primary_angle`` / ``bilateral_shoulder_angles``) — the overlay never
        computes an angle itself, so live and pipeline cannot diverge. Needs
        ``lm2d`` for the joint's pixel position; without it, returns ``None``.
        """
        if not lm3d or not lm2d:
            return None
        joints = measured_joints(self.action)
        if self.action == ActionLabel.SHOULDER_FORWARD_ELEVATION:
            left3, right3 = bilateral_shoulder_angles(lm3d, "3d")
            left2, right2 = bilateral_shoulder_angles(lm2d, "2d")
            angles_3d = {"LEFT_SHOULDER": left3, "RIGHT_SHOULDER": right3}
            angles_2d = {"LEFT_SHOULDER": left2, "RIGHT_SHOULDER": right2}
        else:
            angles_3d = {joints[0]: primary_angle(self.action, lm3d, "3d")}
            angles_2d = {joints[0]: primary_angle(self.action, lm2d, "2d")}
        readouts: list[JointReadout] = []
        for joint in joints:
            position = lm2d.get(joint)
            if position is None:
                continue
            readouts.append(
                JointReadout(
                    joint=joint,
                    x=float(position[0]),
                    y=float(position[1]),
                    angle_2d=angles_2d.get(joint),
                    angle_3d=angles_3d.get(joint),
                )
            )
        return readouts or None

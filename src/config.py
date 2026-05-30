"""Configuration loading for Posture Correction.

All numeric thresholds live in ``config/default.yaml``; this module only reads
and validates them into typed settings. No metric thresholds are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models import ActionLabel


# ---------------------------------------------------------------------------
# settings dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PoseSettings:
    enabled: bool
    min_detection_confidence: float
    min_tracking_confidence: float
    min_landmark_visibility: float
    min_landmark_presence: float
    backend_preference: str
    task_model_path: Path | None
    basis: str                # kinematics: ROM / phase-ROM / rep-segmentation
    compensation_basis: str   # compensation evaluator + calibration baseline
    symmetry_basis: str       # symmetry evaluator (left/right comparison)


@dataclass(slots=True)
class SignalSettings:
    median_window: int
    viz_ema_alpha: float


@dataclass(slots=True)
class PhaseTrackerSettings:
    """Every numeric input to the single state machine (decisions D3, D5)."""

    neutral_confirm_frames: int
    neutral_exit_delta_deg: float
    min_ascent_frames: int
    min_peak_frames: int
    min_descent_frames: int
    peak_confirm_frames: int
    missing_grace_frames: int
    hysteresis_band_deg: float
    pause_epsilon_deg: float
    peak_tolerance_deg: float
    neutral_max_deg: float
    min_cooldown_s: float
    max_baseline_deg: float
    min_dynamic_rom_deg: float
    min_frames: int


@dataclass(slots=True)
class PhaseRomSettings:
    phase_rom_ratio: float


@dataclass(slots=True)
class SymmetrySettings:
    # Amplitude gate — clinical Limb Symmetry Index: |L-R| / mean(L,R) * 100.
    lsi_max_pct: float             # LSI at/below this passes
    lsi_score_best_pct: float      # LSI at/below this scores 100
    lsi_score_zero_pct: float      # LSI at/above this scores 0
    # Onset-timing gate — |left movement-onset - right movement-onset| seconds,
    # detected directly on rep.samples (primary_angle = max(L,R) puts both arms'
    # rises inside the rep window). The pre_rep_lookback_s buffer is legacy.
    onset_diff_max_s: float        # onset gap at/below this passes
    onset_diff_score_best_s: float # gap at/below this scores 100
    onset_diff_score_zero_s: float # gap at/above this scores 0
    onset_velocity_threshold_deg_s: float  # angular speed marking "arm started moving"
    onset_confirm_frames: int      # consecutive over-threshold frames to confirm onset
    pre_rep_lookback_s: float      # how far before rep onset the buffer is scanned


@dataclass(slots=True)
class BaselineSettings:
    """Session-start resting-posture calibration + its quality-control limits."""

    countdown_seconds: float
    target_frames: int
    max_trunk_tilt_spread_deg: float
    max_shoulder_offset_spread_ratio: float


@dataclass(slots=True)
class CompensationSettings:
    trunk_tilt_threshold_deg: float
    shoulder_elevation_threshold_ratio: float
    # Setting-phase upper bound (degrees of primary-angle rise from rep
    # baseline) below which the SE channel is evaluated for shoulder actions.
    # Past this, scapulohumeral rhythm makes the proxy unreliable -> not scored.
    shoulder_elevation_setting_phase_max_deg: float
    min_persist_seconds: float                       # trunk channel
    shoulder_elevation_min_persist_seconds: float
    min_valid_frame_fraction: float
    trunk_tilt_score_zero_deg: float
    shoulder_elevation_score_zero_ratio: float
    failed_score_cap: float
    baseline: BaselineSettings


@dataclass(slots=True)
class PathsSettings:
    rom_reference_path: Path
    session_output_dir: Path


@dataclass(slots=True)
class LiveSymmetryDisplaySettings:
    history_maxlen: int
    chart_y_max_deg: float


@dataclass(slots=True)
class LiveSettings:
    symmetry: LiveSymmetryDisplaySettings


@dataclass(slots=True)
class AppSettings:
    project_name: str
    supported_actions: tuple[ActionLabel, ...]
    pose: PoseSettings
    signal: SignalSettings
    phase_tracker: PhaseTrackerSettings
    phase_rom: PhaseRomSettings
    symmetry: SymmetrySettings
    compensation: CompensationSettings
    paths: PathsSettings
    live: LiveSettings
    project_root: Path


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_basis(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"2d", "3d"}:
        raise ValueError(f"pose.basis must be '2d' or '3d', got {value!r}")
    return text


def load_settings(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> AppSettings:
    """Load settings from ``config/default.yaml``, optionally merging an override."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    payload = yaml.safe_load((root / "config" / "default.yaml").read_text(encoding="utf-8"))
    if config_path is not None:
        override = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        payload = _deep_merge(payload, override)

    pose = payload["pose"]
    signal = payload["signal"]
    tracker = payload["phase_tracker"]
    phase_rom = payload["phase_rom"]
    symmetry = payload["symmetry"]
    compensation = payload["compensation"]
    paths = payload["paths"]
    live = payload["live"]

    return AppSettings(
        project_name=str(payload["project_name"]),
        supported_actions=tuple(
            ActionLabel.from_value(value) for value in payload["supported_actions"]
        ),
        pose=PoseSettings(
            enabled=bool(pose["enabled"]),
            min_detection_confidence=float(pose["min_detection_confidence"]),
            min_tracking_confidence=float(pose["min_tracking_confidence"]),
            min_landmark_visibility=float(pose["min_landmark_visibility"]),
            min_landmark_presence=float(pose["min_landmark_presence"]),
            backend_preference=str(pose["backend_preference"]),
            task_model_path=(
                (root / pose["task_model_path"]).resolve()
                if pose.get("task_model_path")
                else None
            ),
            basis=_validate_basis(pose["basis"]),
            compensation_basis=_validate_basis(pose["compensation_basis"]),
            symmetry_basis=_validate_basis(pose["symmetry_basis"]),
        ),
        signal=SignalSettings(
            median_window=int(signal["median_window"]),
            viz_ema_alpha=float(signal["viz_ema_alpha"]),
        ),
        phase_tracker=PhaseTrackerSettings(
            neutral_confirm_frames=int(tracker["neutral_confirm_frames"]),
            neutral_exit_delta_deg=float(tracker["neutral_exit_delta_deg"]),
            min_ascent_frames=int(tracker["min_ascent_frames"]),
            min_peak_frames=int(tracker["min_peak_frames"]),
            min_descent_frames=int(tracker["min_descent_frames"]),
            peak_confirm_frames=int(tracker["peak_confirm_frames"]),
            missing_grace_frames=int(tracker["missing_grace_frames"]),
            hysteresis_band_deg=float(tracker["hysteresis_band_deg"]),
            pause_epsilon_deg=float(tracker["pause_epsilon_deg"]),
            peak_tolerance_deg=float(tracker["peak_tolerance_deg"]),
            neutral_max_deg=float(tracker["neutral_max_deg"]),
            min_cooldown_s=float(tracker["min_cooldown_s"]),
            max_baseline_deg=float(tracker["max_baseline_deg"]),
            min_dynamic_rom_deg=float(tracker["min_dynamic_rom_deg"]),
            min_frames=int(tracker["min_frames"]),
        ),
        phase_rom=PhaseRomSettings(
            phase_rom_ratio=float(phase_rom["phase_rom_ratio"]),
        ),
        symmetry=SymmetrySettings(
            lsi_max_pct=float(symmetry["lsi_max_pct"]),
            lsi_score_best_pct=float(symmetry["lsi_score_best_pct"]),
            lsi_score_zero_pct=float(symmetry["lsi_score_zero_pct"]),
            onset_diff_max_s=float(symmetry["onset_diff_max_s"]),
            onset_diff_score_best_s=float(symmetry["onset_diff_score_best_s"]),
            onset_diff_score_zero_s=float(symmetry["onset_diff_score_zero_s"]),
            onset_velocity_threshold_deg_s=float(
                symmetry["onset_velocity_threshold_deg_s"]
            ),
            onset_confirm_frames=int(symmetry["onset_confirm_frames"]),
            pre_rep_lookback_s=float(symmetry["pre_rep_lookback_s"]),
        ),
        compensation=CompensationSettings(
            trunk_tilt_threshold_deg=float(compensation["trunk_tilt_threshold_deg"]),
            shoulder_elevation_threshold_ratio=float(
                compensation["shoulder_elevation_threshold_ratio"]
            ),
            shoulder_elevation_setting_phase_max_deg=float(
                compensation["shoulder_elevation_setting_phase_max_deg"]
            ),
            min_persist_seconds=float(compensation["min_persist_seconds"]),
            shoulder_elevation_min_persist_seconds=float(
                compensation["shoulder_elevation_min_persist_seconds"]
            ),
            min_valid_frame_fraction=float(compensation["min_valid_frame_fraction"]),
            trunk_tilt_score_zero_deg=float(compensation["trunk_tilt_score_zero_deg"]),
            shoulder_elevation_score_zero_ratio=float(
                compensation["shoulder_elevation_score_zero_ratio"]
            ),
            failed_score_cap=float(compensation["failed_score_cap"]),
            baseline=BaselineSettings(
                countdown_seconds=float(compensation["baseline"]["countdown_seconds"]),
                target_frames=int(compensation["baseline"]["target_frames"]),
                max_trunk_tilt_spread_deg=float(
                    compensation["baseline"]["max_trunk_tilt_spread_deg"]
                ),
                max_shoulder_offset_spread_ratio=float(
                    compensation["baseline"]["max_shoulder_offset_spread_ratio"]
                ),
            ),
        ),
        paths=PathsSettings(
            rom_reference_path=(root / paths["rom_reference_path"]).resolve(),
            session_output_dir=(root / paths["session_output_dir"]).resolve(),
        ),
        live=LiveSettings(
            symmetry=LiveSymmetryDisplaySettings(
                history_maxlen=int(live["symmetry"]["history_maxlen"]),
                chart_y_max_deg=float(live["symmetry"]["chart_y_max_deg"]),
            ),
        ),
        project_root=root,
    )

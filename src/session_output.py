"""Session output assembly — turn completed-rep snapshots into a JSON payload.

Pure stdlib, no metric logic: every value comes verbatim from the pipeline's
own ``MetricResult.to_dict()`` / ``Feedback.to_dict()``. The scored output
never reads ``rep_debug``; the one exception is :func:`compensation_debug_to_dict`,
an opt-in diagnostic serializer (used only behind ``app.py --debug``) that
dumps the per-frame compensation profile for offline inspection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_NAME = "posture-correction"


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def compensation_debug_to_dict(payload: Any) -> dict[str, Any]:
    """Serialize a ``CompensationDebugPayload`` for the ``--debug`` JSON dump.

    Diagnostic only — never part of scoring. Reads attributes by name (no type
    import) so the scoring/output layers stay decoupled from the payload type.
    """
    def _r(values: list[float]) -> list[float]:
        return [round(v, 4) for v in values]

    return {
        "rep_baseline_angle_deg": round(payload.rep_baseline_angle_deg, 2),
        "setting_phase_max_deg": payload.setting_phase_max_deg,
        "trunk_tilt_threshold_deg": payload.trunk_tilt_threshold_deg,
        "shoulder_elevation_threshold_ratio": payload.shoulder_elevation_threshold_ratio,
        "trunk_tilt_detected": payload.trunk_tilt_detected,
        "shoulder_elevation_detected": payload.shoulder_elevation_detected,
        "basis_used": payload.basis_used,
        "timestamps_s": _r(payload.timestamps_s),
        "primary_angle_deg": _r(payload.primary_angle_deg),
        "trunk_tilt_deviation_deg": _r(payload.trunk_tilt_deviation_deg),
        "shoulder_elevation_ratio_left": _r(payload.shoulder_elevation_ratio_left),
        "shoulder_elevation_ratio_right": _r(payload.shoulder_elevation_ratio_right),
    }


def rep_to_dict(rep_index: int, snapshot: Any) -> dict[str, Any]:
    """One completed-rep ``SessionSnapshot`` -> a serialisable rep record."""
    return {
        "rep_index": rep_index,
        "overall_score": _round(snapshot.overall_score),
        "metrics": [result.to_dict() for result in snapshot.rep_results],
        "feedback": snapshot.feedback.to_dict() if snapshot.feedback is not None else None,
    }


def summarize(rep_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Session-level roll-up: rep count, mean score, per-metric pass tally."""
    scores = [r["overall_score"] for r in rep_dicts if r["overall_score"] is not None]
    tally: dict[str, dict[str, int]] = {}
    for rep in rep_dicts:
        for metric in rep["metrics"]:
            counts = tally.setdefault(
                metric["name"], {"passed": 0, "failed": 0, "other": 0}
            )
            if metric["passed"] is True:
                counts["passed"] += 1
            elif metric["passed"] is False:
                counts["failed"] += 1
            else:
                counts["other"] += 1
    return {
        "rep_count": len(rep_dicts),
        "mean_overall_score": (
            round(sum(scores) / len(scores), 2) if scores else None
        ),
        "metric_pass_counts": tally,
    }


def build_session_payload(
    action_value: str,
    source: str,
    processed_frames: int,
    rep_dicts: list[dict[str, Any]],
    is_complete: bool,
    pose_basis: str = "2d",
    compensation_basis: str = "3d",
) -> dict[str, Any]:
    """Assemble the full session payload written to JSON.

    ``pose_basis`` records the kinematics basis used for ROM / phase-ROM /
    symmetry. ``compensation_basis`` records the basis used for the
    compensation evaluator and its calibration baseline.
    """
    return {
        "project": PROJECT_NAME,
        "action": action_value,
        "source": source,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_complete": is_complete,
        "processed_frames": processed_frames,
        "pose_basis": pose_basis,
        "compensation_basis": compensation_basis,
        "rep_count": len(rep_dicts),
        "reps": rep_dicts,
        "summary": summarize(rep_dicts),
    }


def resolve_output_path(
    output_dir: Path,
    action_value: str,
    explicit: Path | None,
) -> Path:
    """Explicit path wins; otherwise ``output_dir/session_<timestamp>_<action>.json``."""
    if explicit is not None:
        return Path(explicit)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"session_{stamp}_{action_value}.json"


def write_session_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

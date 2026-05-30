"""Posture Correction production runner.

Processes a video file or camera through the shared ``LiveSession`` pipeline
for one pre-selected action, writes a per-rep evaluation JSON to the configured
``session_output_dir`` and prints a console summary. Headless — no window; use
``live.apps.*`` to watch a session live.

Run via the installed console script::

    posture-correction --action shoulder_flexion_right --src clip.mp4
    posture-correction --action shoulder_flexion_right --src 0      # camera
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from config import AppSettings, load_settings
from live.app_runner import mirror_snapshot
from live.overlay_renderer import LiveOverlayRenderer
from live.session import LiveSession
from models import ActionLabel
from pipeline.pose_detector import build_pose_tracker
from session_output import (
    build_session_payload,
    compensation_debug_to_dict,
    rep_to_dict,
    resolve_output_path,
    write_session_json,
)

_DEFAULT_FPS = 30.0


@dataclass(slots=True)
class RunRequest:
    source: str                       # camera index (digits) or video file path
    action: ActionLabel
    output_path: Path | None = None   # explicit JSON path; None -> auto-named
    debug: bool = False               # attach per-rep compensation_debug profile


def _metric_line(rep: dict[str, Any]) -> str:
    parts = []
    for metric in rep["metrics"]:
        passed = metric["passed"]
        mark = "pass" if passed is True else "fail" if passed is False else metric["status"]
        parts.append(f"{metric['name']}:{mark}")
    return "  ".join(parts)


def _save_session_payload(
    settings: AppSettings,
    action_value: str,
    source: str,
    frame_index: int,
    rep_dicts: list[dict[str, Any]],
    is_complete: bool,
    explicit_output_path: Path | None,
    baseline_unsteady: bool,
) -> tuple[Path, dict[str, Any]]:
    """Build, write and print one session payload; return (path, payload).

    Shared by the session-end save and the camera-mode ``R`` (recalibrate)
    handler so the JSON shape, filename rules and warning output never drift
    between the two paths.
    """
    payload = build_session_payload(
        action_value, source, frame_index, rep_dicts, is_complete,
        settings.pose.basis, settings.pose.compensation_basis,
    )
    output_path = resolve_output_path(
        settings.paths.session_output_dir, action_value, explicit_output_path
    )
    write_session_json(output_path, payload)
    if baseline_unsteady:
        print(
            "warning: resting-posture calibration was unsteady - "
            "compensation results for this session may be unreliable"
        )
    _print_summary(payload, output_path)
    return output_path, payload


def run_session(request: RunRequest, settings: AppSettings) -> dict[str, Any]:
    """Run one session end-to-end; write the JSON and print a summary.

    Camera mode (``--src`` digits) opens a live mirror window — ``q`` / ``ESC``
    stops it. Video mode is headless. The pose pipeline always runs on the TRUE
    frame so the action's left/right matches the body; only the camera window's
    image (and its body-anchored overlay) is mirrored.
    """
    import cv2

    is_camera = request.source.isdigit()
    source: int | str = int(request.source) if is_camera else request.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: cannot open source {request.source!r}")
        return {"ok": False, "source": request.source}

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # 0 / negative / NaN
        fps = _DEFAULT_FPS

    pose_tracker = build_pose_tracker(settings.pose)
    # Camera: manual resting-posture calibration (press R, countdown, collect).
    # Video: headless auto-calibration from the leading valid frames.
    session = LiveSession(
        settings,
        pose_tracker,
        request.action,
        calibration="manual" if is_camera else "auto",
    )
    renderer = LiveOverlayRenderer() if is_camera else None
    window = f"Posture Correction - {request.action.value}" if is_camera else ""
    rep_dicts: list[dict[str, Any]] = []
    frame_index = 0
    is_complete = False
    start = perf_counter()

    print(
        f"running {request.action.value} on {request.source!r} "
        f"({'camera' if is_camera else 'video'}); "
        f"{'press R to calibrate, q / ESC to stop' if is_camera else 'Ctrl-C to stop'}"
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                is_complete = True
                break
            timestamp = (perf_counter() - start) if is_camera else (frame_index / fps)
            snapshot = session.process_frame(frame, timestamp)
            if snapshot.rep_results is not None:
                rep = rep_to_dict(len(rep_dicts) + 1, snapshot)
                if request.debug and snapshot.rep_debug is not None:
                    comp_debug = snapshot.rep_debug.get("compensation")
                    if comp_debug is not None:
                        rep["compensation_debug"] = compensation_debug_to_dict(comp_debug)
                rep_dicts.append(rep)
                print(
                    f"[rep {rep['rep_index']}] overall="
                    f"{rep['overall_score']}  {_metric_line(rep)}"
                )
            frame_index += 1
            if is_camera and renderer is not None:
                display = cv2.flip(frame, 1)
                renderer.render(
                    display, mirror_snapshot(snapshot, display.shape[1]), len(rep_dicts)
                )
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break  # user stop -> is_complete stays False
                if key == ord("r"):
                    # Recalibrate: the rep_dicts collected so far ARE a valid
                    # session - save them under an auto-named file (never the
                    # caller's -o; that one is reserved for the final session
                    # at exit, so multiple R presses cannot overwrite it),
                    # then reset the rep tally and timeline so the next
                    # sub-session is independent.
                    if rep_dicts:
                        _save_session_payload(
                            settings,
                            request.action.value,
                            request.source,
                            frame_index,
                            rep_dicts,
                            is_complete=True,
                            explicit_output_path=None,
                            baseline_unsteady=session.baseline_unsteady,
                        )
                        rep_dicts.clear()
                        # Reset the rep counter shown in the live overlay so
                        # the next sub-session starts from rep 1.
                        # (len(rep_dicts) is what the overlay reads below.)
                    frame_index = 0
                    start = perf_counter()
                    session.start_calibration()
    except KeyboardInterrupt:
        print("\ninterrupted - finishing session")
    finally:
        capture.release()
        if is_camera:
            cv2.destroyAllWindows()
        pose_tracker.close()

    output_path, payload = _save_session_payload(
        settings,
        request.action.value,
        request.source,
        frame_index,
        rep_dicts,
        is_complete,
        request.output_path,
        session.baseline_unsteady,
    )
    return {
        "ok": True,
        "source": request.source,
        "output_path": str(output_path),
        "processed_frames": frame_index,
        "rep_count": len(rep_dicts),
        "summary": payload["summary"],
    }


def _print_summary(payload: dict[str, Any], output_path: Path) -> None:
    summary = payload["summary"]
    print("-" * 48)
    print(f"action          : {payload['action']}")
    print(f"processed frames: {payload['processed_frames']}")
    print(f"complete        : {payload['is_complete']}")
    print(f"reps            : {summary['rep_count']}")
    print(f"mean score      : {summary['mean_overall_score']}")
    for name, counts in summary["metric_pass_counts"].items():
        print(
            f"  {name:<12} pass={counts['passed']} "
            f"fail={counts['failed']} other={counts['other']}"
        )
    print(f"output          : {output_path}")
    print("-" * 48)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="posture-correction",
        description="Evaluate a posture-correction session from a video file or camera.",
    )
    parser.add_argument(
        "--action", required=True, choices=ActionLabel.choices(),
        help="the exercise being performed (no action classifier)",
    )
    parser.add_argument(
        "--src", default="0",
        help="camera index (digits) or video file path; default 0",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="explicit JSON output path; default auto-named under session_output_dir",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="attach a per-frame compensation_debug profile to each rep in the JSON",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    request = RunRequest(
        source=args.src,
        action=ActionLabel.from_value(args.action),
        output_path=Path(args.output) if args.output else None,
        debug=args.debug,
    )
    result = run_session(request, settings)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

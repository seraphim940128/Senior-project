"""Webcam loop for the live demo apps.

Owns the camera and the cv2 window; per frame it calls ``LiveSession`` and
hands the snapshot to a ``LiveMetricApp`` for rendering. ``cv2`` is imported
lazily so this module imports without the optional ``vision`` extra.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from config import AppSettings
from live.session import LiveSession
from models import ActionLabel

if TYPE_CHECKING:
    from live.session import SessionSnapshot
    from pipeline.pose_detector import PoseTracker


def _cv2():
    """Lazy ``cv2`` import — keeps this module importable without opencv-python."""
    import cv2

    return cv2


@dataclass(slots=True)
class RunnerConfig:
    camera_index: int = 0
    window_name: str = "Posture Correction"
    flip_horizontal: bool = True
    calibration: str = "off"          # off | manual — see LiveSession


class LiveMetricApp(Protocol):
    """Contract a demo app implements; the runner calls only this surface.

    ``on_frame`` draws the main camera window; ``on_charts`` draws the separate
    chart window onto a dark canvas. Both are called every frame.
    """

    title: str
    action: ActionLabel

    def on_frame(self, image, snapshot: "SessionSnapshot", rep_index: int): ...

    def on_charts(self, canvas, snapshot: "SessionSnapshot", rep_index: int): ...

    def on_rep(self, snapshot: "SessionSnapshot") -> None: ...


def mirror_snapshot(snapshot: "SessionSnapshot", width: int) -> "SessionSnapshot":
    """Mirror the body-anchored pixel coords for a horizontally-flipped display.

    The pipeline already ran on the TRUE frame, so left/right are correct; this
    only flips the *drawing* positions (``x -> width - x``) of ``pose_xy`` and
    ``joint_readouts`` so the skeleton / joint labels line up with the mirrored
    image. Corner-anchored panels (text, progress bar) are not coordinates and
    are left untouched.
    """
    pose_xy = snapshot.pose_xy
    if pose_xy is not None:
        pose_xy = {name: (width - x, y) for name, (x, y) in pose_xy.items()}
    readouts = snapshot.joint_readouts
    if readouts is not None:
        readouts = [replace(r, x=width - r.x) for r in readouts]
    return replace(snapshot, pose_xy=pose_xy, joint_readouts=readouts)


def run_live_app(
    app: LiveMetricApp,
    settings: AppSettings,
    pose_tracker: "PoseTracker",
    runner: RunnerConfig | None = None,
) -> None:
    """Drive ``app`` from the webcam in two windows until the user quits.

    Main window = camera image + overlay; second window = charts. ``q`` / ``ESC``
    quit; ``r`` resets the session and the rep counter mid-run.

    The pose pipeline always runs on the TRUE (un-flipped) frame so the action's
    left/right matches the body; when ``flip_horizontal`` is set, only the
    displayed image and the body-anchored overlay coords are mirrored.
    """
    cv = _cv2()
    runner = runner or RunnerConfig()
    # Set up cleanup state BEFORE acquiring any resource so a failure during
    # camera open still releases pose_tracker (the caller passed ownership
    # to us and would otherwise leak it).
    capture = None
    try:
        capture = cv.VideoCapture(runner.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open camera index {runner.camera_index}")

        main_window = f"{runner.window_name} - {app.title}"
        chart_window = f"{runner.window_name} - {app.title} - charts"
        session = LiveSession(
            settings, pose_tracker, app.action, calibration=runner.calibration
        )
        rep_index = 0
        start = perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # process the TRUE frame - left/right must match the real body
            snapshot = session.process_frame(frame, perf_counter() - start)
            if snapshot.rep_results is not None:
                rep_index += 1
                app.on_rep(snapshot)
            # mirror only for display (natural "look in a mirror" view)
            if runner.flip_horizontal:
                display = cv.flip(frame, 1)
                main_snapshot = mirror_snapshot(snapshot, display.shape[1])
            else:
                display = frame
                main_snapshot = snapshot
            main_image = app.on_frame(display, main_snapshot, rep_index)
            # dark chart canvas - derived from the camera ndarray, no numpy import
            canvas = display.copy()
            canvas[:] = 18
            chart_image = app.on_charts(canvas, snapshot, rep_index)
            cv.imshow(main_window, main_image)
            cv.imshow(chart_window, chart_image)
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                # R restarts: in a calibrated app it (re)runs the resting-posture
                # calibration; otherwise it just resets the session.
                if runner.calibration == "off":
                    session.reset()
                else:
                    session.start_calibration()
                rep_index = 0
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("screenshots", exist_ok=True)
                prefix = f"screenshots/{app.action}_{ts}"
                cv.imwrite(f"{prefix}_chart.png", chart_image)
                cv.imwrite(f"{prefix}_main.png", main_image)
                print(f"[screenshot] saved {prefix}_*.png")
    finally:
        if capture is not None:
            capture.release()
        cv.destroyAllWindows()
        pose_tracker.close()


def build_and_run(action: ActionLabel, runner: RunnerConfig | None = None) -> None:
    """Convenience entry point: load settings, build the pose tracker, run with
    the default standard-panel app.
    """
    from config import load_settings
    from pipeline.pose_detector import build_pose_tracker
    from live.overlay_renderer import LiveOverlayRenderer

    settings = load_settings()
    pose_tracker = build_pose_tracker(settings.pose)
    app = DefaultLiveApp(action=action, renderer=LiveOverlayRenderer())
    run_live_app(app, settings, pose_tracker, runner)


@dataclass(slots=True)
class DefaultLiveApp:
    """Minimal LiveMetricApp: the standard overlay panel, no metric extras."""

    action: ActionLabel
    renderer: object
    title: str = "live"

    def on_frame(self, image, snapshot, rep_index):
        return self.renderer.render(image, snapshot, rep_index)

    def on_charts(self, canvas, snapshot, rep_index):
        return canvas

    def on_rep(self, snapshot) -> None:
        return None

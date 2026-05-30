"""Live compensation validation app — trunk-tilt + shoulder-elevation traces.

Independent demo program: only wires ``LiveSession`` + the overlay renderer.
Run ``python -m live.apps.live_compensation [action]`` (with the ``vision``
extra and a camera). It starts in the resting-posture calibration flow — the
overlay shows ``Stand ready, then press R``; press R, hold still through the
countdown and the 30-frame capture, then detection begins.

Opens two windows — main (camera + skeleton + live compensation readout) and
charts. The trunk / shoulder curves come from the compensation evaluator's
debug payload; this file does no compensation math.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field

from config import load_settings
from live.app_runner import RunnerConfig, run_live_app
from live.overlay_renderer import (
    ChartSeries,
    ChartStatus,
    LiveOverlayRenderer,
    draw_chart,
)
from models import ActionLabel
from pipeline.pose_detector import build_pose_tracker

_TOP_RECT = (20, 20, 600, 140)
_MID_RECT = (20, 170, 600, 140)
_BOTTOM_RECT = (20, 320, 600, 140)
_ANGLE_COLOR = (0, 220, 255)
_TRUNK_COLOR = (0, 200, 255)
_LEFT_COLOR = (0, 255, 255)
_RIGHT_COLOR = (0, 180, 0)
_THRESHOLD_COLOR = (60, 60, 230)
_ANGLE_Y_MAX_DEG = 180.0


def _channel_status(result, detected: bool) -> ChartStatus:
    """Per-channel chart badge: each channel reflects its OWN detection.

    The two last-rep charts (trunk, shoulder) each show whether THAT channel
    detected compensation, rather than the metric's overall pass/fail — so a
    shoulder hike lights up the shoulder chart, not the trunk chart. ``result``
    supplies the reliability gate (an ``unreliable`` rep cannot claim a result);
    ``detected`` is the channel's own flag from the debug payload.
    """
    if result is None:
        return ChartStatus("pending", "waiting for rep")
    if result.status != "ok":
        return ChartStatus("pending", result.status)
    if detected:
        return ChartStatus("failure", "detected")
    return ChartStatus("success", "clear")


def _flat(value: float, length: int) -> list[float]:
    """A constant series of ``length`` samples — used to draw a threshold line."""
    return [value] * length


@dataclass(slots=True)
class LiveCompensationApp:
    action: ActionLabel
    renderer: LiveOverlayRenderer
    history: deque = field(default_factory=lambda: deque(maxlen=180))
    last_payload: object = None
    last_result: object = None
    title: str = "Compensation"

    def on_frame(self, image, snapshot, rep_index):
        """Main window: camera + skeleton + live compensation readout + panel."""
        self.renderer.render(image, snapshot, rep_index)
        return image

    def on_charts(self, canvas, snapshot, rep_index):
        """Chart window: rolling angle + last rep's trunk / shoulder curves."""
        if snapshot.raw_angle is not None:
            self.history.append(snapshot.raw_angle)
        if snapshot.rep_debug is not None:
            payload = snapshot.rep_debug.get("compensation")
            if payload is not None:
                self.last_payload = payload

        draw_chart(
            canvas, _TOP_RECT, "Primary Angle (rolling)",
            [ChartSeries("angle", list(self.history), _ANGLE_COLOR)],
            y_bounds=(0.0, _ANGLE_Y_MAX_DEG),
        )

        trunk = list(getattr(self.last_payload, "trunk_tilt_deviation_deg", []) or [])
        trunk_series = [ChartSeries("trunk tilt dev", trunk, _TRUNK_COLOR)]
        if trunk:
            thr = getattr(self.last_payload, "trunk_tilt_threshold_deg", 0.0)
            trunk_series.append(
                ChartSeries("threshold", _flat(thr, len(trunk)), _THRESHOLD_COLOR)
            )
        draw_chart(
            canvas, _MID_RECT, "Trunk Lateral Tilt (last rep)", trunk_series,
            status=_channel_status(
                self.last_result,
                getattr(self.last_payload, "trunk_tilt_detected", False),
            ),
        )

        left = list(getattr(self.last_payload, "shoulder_elevation_ratio_left", []) or [])
        right = list(getattr(self.last_payload, "shoulder_elevation_ratio_right", []) or [])
        shoulder_series = [
            ChartSeries("left", left, _LEFT_COLOR),
            ChartSeries("right", right, _RIGHT_COLOR),
        ]
        span = max(len(left), len(right))
        if span:
            thr = getattr(self.last_payload, "shoulder_elevation_threshold_ratio", 0.0)
            shoulder_series.append(
                ChartSeries("threshold", _flat(thr, span), _THRESHOLD_COLOR)
            )
        draw_chart(
            canvas, _BOTTOM_RECT, "Shoulder Elevation Ratio (last rep)", shoulder_series,
            status=_channel_status(
                self.last_result,
                getattr(self.last_payload, "shoulder_elevation_detected", False),
            ),
            marker_index=getattr(
                self.last_payload, "shoulder_setting_phase_end_idx", -1
            ),
        )
        return canvas

    def on_rep(self, snapshot) -> None:
        result = next(
            (r for r in snapshot.rep_results if r.name == "compensation"), None
        )
        if result is not None:
            self.last_result = result
            print(
                f"[compensation] status={result.status} passed={result.passed} "
                f"value={result.primary_value} score={result.score} "
                f"detail={result.detail}"
            )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    action = (
        ActionLabel.from_value(argv[0]) if argv else ActionLabel.SHOULDER_ABDUCTION_RIGHT
    )
    app = LiveCompensationApp(action=action, renderer=LiveOverlayRenderer())
    run_live_app(
        app,
        settings,
        build_pose_tracker(settings.pose),
        RunnerConfig(calibration="manual"),
    )


if __name__ == "__main__":
    main()

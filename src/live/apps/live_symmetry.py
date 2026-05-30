"""Live symmetry validation app — left/right shoulder trace comparison.

Independent demo program: only wires ``LiveSession`` + the overlay renderer.
Run ``python -m live.apps.live_symmetry`` (with the ``vision`` extra and a
camera). Opens two windows — main (camera + skeleton + left/right shoulder
angles) and charts. The left/right waveforms come from the symmetry evaluator's
debug payload; this file does no symmetry math.

In the chart window the top chart is a per-frame rolling angle trace; the lower
chart holds the most recent completed rep's left/right curves and persists
until the next rep replaces them — per-frame left/right values are not exposed
by the session, so the last rep is kept rather than recomputed here.
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

_TOP_RECT = (20, 20, 600, 210)
_BOTTOM_RECT = (20, 245, 600, 215)
_ANGLE_COLOR = (0, 220, 255)
_LEFT_COLOR = (0, 255, 255)
_RIGHT_COLOR = (0, 180, 0)


def _fmt(value) -> str:
    """Compact float formatting for the diagnostic console line; passes None through."""
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def _chart_status(result) -> ChartStatus:
    """Map the latest symmetry MetricResult to a chart status badge."""
    if result is None:
        return ChartStatus("pending", "waiting for rep")
    if result.passed is True:
        return ChartStatus("success", f"score {result.score:.0f}")
    if result.passed is False:
        return ChartStatus("failure", f"score {result.score:.0f}")
    return ChartStatus("pending", result.status)


@dataclass(slots=True)
class LiveSymmetryApp:
    action: ActionLabel
    renderer: LiveOverlayRenderer
    chart_y_max_deg: float
    history: deque = field(default_factory=lambda: deque(maxlen=180))
    last_payload: object = None
    last_result: object = None
    title: str = "Symmetry"

    def on_frame(self, image, snapshot, rep_index):
        """Main window: camera + skeleton + left/right shoulder angles + panel."""
        self.renderer.render(image, snapshot, rep_index)
        return image

    def on_charts(self, canvas, snapshot, rep_index):
        """Chart window: rolling elevation chart + last rep's left/right curves."""
        if snapshot.raw_angle is not None:
            self.history.append(snapshot.raw_angle)
        if snapshot.rep_debug is not None:
            payload = snapshot.rep_debug.get("symmetry")
            if payload is not None:
                self.last_payload = payload
        draw_chart(
            canvas, _TOP_RECT, "Elevation (rolling)",
            [ChartSeries("angle", list(self.history), _ANGLE_COLOR)],
            y_bounds=(0.0, self.chart_y_max_deg),
        )
        left = list(getattr(self.last_payload, "smoothed_left_deg", []) or [])
        right = list(getattr(self.last_payload, "smoothed_right_deg", []) or [])
        draw_chart(
            canvas, _BOTTOM_RECT, "Last Rep L / R",
            [
                ChartSeries("left", left, _LEFT_COLOR),
                ChartSeries("right", right, _RIGHT_COLOR),
            ],
            y_bounds=(0.0, self.chart_y_max_deg),
            status=_chart_status(self.last_result),
        )
        return canvas

    def on_rep(self, snapshot) -> None:
        result = next((r for r in snapshot.rep_results if r.name == "symmetry"), None)
        if result is not None:
            self.last_result = result
            # Lead with the two GATES (lsi vs lsi_max_pct, onset_diff vs
            # onset_diff_max_s) so it is clear which one passed/failed; the rest
            # (waveform_sim / ncc / peak_lag) are diagnostics only.
            d = result.detail
            print(
                f"[symmetry] status={result.status} passed={result.passed} "
                f"score={_fmt(result.score)} | GATES "
                f"lsi={_fmt(d.get('lsi_pct'))}% "
                f"onset_diff={_fmt(d.get('onset_diff_s'))}s "
                f"(L@{_fmt(d.get('left_onset_s'))} R@{_fmt(d.get('right_onset_s'))}) "
                f"| diag waveform_sim={_fmt(d.get('waveform_similarity'))} "
                f"ncc={_fmt(d.get('waveform_ncc'))} peak_lag={_fmt(d.get('peak_lag_frac'))} "
                f"| L_rom={_fmt(d.get('left_rom_deg'))} R_rom={_fmt(d.get('right_rom_deg'))}"
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Takes no arguments: symmetry is locked to
    ``shoulder_forward_elevation`` (the only bilateral action). Passing any
    argument is an error so users do not silently get a different action
    than they intended.
    """
    args = sys.argv[1:] if argv is None else argv
    if args:
        print(
            "error: live_symmetry takes no arguments "
            "(locked to shoulder_forward_elevation)",
            file=sys.stderr,
        )
        return 2
    settings = load_settings()
    app = LiveSymmetryApp(
        action=ActionLabel.SHOULDER_FORWARD_ELEVATION,
        renderer=LiveOverlayRenderer(),
        chart_y_max_deg=settings.live.symmetry.chart_y_max_deg,
        history=deque(maxlen=settings.live.symmetry.history_maxlen),
    )
    run_live_app(app, settings, build_pose_tracker(settings.pose), RunnerConfig())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

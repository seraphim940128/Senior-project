"""Live phase-ROM validation app — ascent / peak / descent breakdown.

Independent demo program: only wires ``LiveSession`` + the overlay renderer.
Run ``python -m live.apps.live_phase_rom [action]`` (with the ``vision`` extra
and a camera). Opens two windows — main (camera + skeleton + angles + phase
counters) and charts (primary-angle chart + phase timeline). The phase
segmentation is the PhaseTracker's; this file only displays it.
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
    OverlayStyle,
    draw_chart,
    draw_phase_legend,
    draw_phase_timeline,
    draw_text,
)
from models import ActionLabel
from pipeline.pose_detector import build_pose_tracker

_CHART_RECT = (20, 20, 600, 285)
_TIMELINE_RECT = (20, 315, 600, 95)
_LEGEND_ORIGIN = (32, 445)
_CHART_Y_MAX_DEG = 180.0
_ANGLE_COLOR = (0, 220, 255)


def _chart_status(result) -> ChartStatus:
    """Map the latest phase-ROM MetricResult to a chart status badge."""
    if result is None:
        return ChartStatus("pending", "waiting for rep")
    if result.passed is True:
        return ChartStatus("success", f"score {result.score:.0f}")
    if result.passed is False:
        return ChartStatus("failure", f"score {result.score:.0f}")
    return ChartStatus("pending", result.status)


@dataclass(slots=True)
class LivePhaseRomApp:
    action: ActionLabel
    renderer: LiveOverlayRenderer
    history: deque = field(default_factory=lambda: deque(maxlen=180))
    phase_history: deque = field(default_factory=lambda: deque(maxlen=180))
    last_result: object = None
    title: str = "Phase ROM"

    def on_frame(self, image, snapshot, rep_index):
        """Main window: camera + skeleton + joint angles + phase counters."""
        self.renderer.render(image, snapshot, rep_index)
        phase = snapshot.phase
        style = OverlayStyle()
        draw_text(
            image,
            f"ascent={phase.ascent_frames} peak={phase.peak_frames} "
            f"descent={phase.descent_frames}",
            style.origin_x, 300, style,
        )
        return image

    def on_charts(self, canvas, snapshot, rep_index):
        """Chart window: rolling primary-angle chart + phase timeline."""
        if snapshot.raw_angle is not None:
            self.history.append(snapshot.raw_angle)
        self.phase_history.append(snapshot.phase.current_phase)
        draw_chart(
            canvas, _CHART_RECT, "Primary Angle",
            [ChartSeries("angle", list(self.history), _ANGLE_COLOR)],
            y_bounds=(0.0, _CHART_Y_MAX_DEG),
            status=_chart_status(self.last_result),
        )
        draw_phase_timeline(canvas, _TIMELINE_RECT, list(self.phase_history))
        draw_phase_legend(canvas, _LEGEND_ORIGIN)
        return canvas

    def on_rep(self, snapshot) -> None:
        result = next((r for r in snapshot.rep_results if r.name == "phase_rom"), None)
        if result is not None:
            self.last_result = result
            print(
                f"[phase_rom] status={result.status} passed={result.passed} "
                f"detail={result.detail}"
            )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    action = (
        ActionLabel.from_value(argv[0]) if argv else ActionLabel.SHOULDER_FLEXION_RIGHT
    )
    app = LivePhaseRomApp(action=action, renderer=LiveOverlayRenderer())
    run_live_app(app, settings, build_pose_tracker(settings.pose), RunnerConfig())


if __name__ == "__main__":
    main()

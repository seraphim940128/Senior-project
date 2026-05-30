"""Live ROM validation app — webcam ROM bar against the per-action minimum.

Independent demo program: only wires ``LiveSession`` + the overlay renderer.
Run ``python -m live.apps.live_rom [action]`` (with the ``vision`` extra and a
camera). Opens two windows — main (camera + skeleton + angles) and charts.
The metric logic lives in the pipeline; this file does no metric math.
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
    draw_progress_bar,
)
from models import ActionLabel
from pipeline.evaluators.rom import load_rom_references
from pipeline.pose_detector import build_pose_tracker

_CHART_RECT = (20, 20, 600, 440)
_CHART_Y_MAX_DEG = 180.0
_ANGLE_COLOR = (0, 220, 255)


def _chart_status(result) -> ChartStatus:
    """Map the latest ROM MetricResult to a chart status badge."""
    if result is None:
        return ChartStatus("pending", "waiting for rep")
    if result.passed is True:
        return ChartStatus("success", f"score {result.score:.0f}")
    if result.passed is False:
        return ChartStatus("failure", f"score {result.score:.0f}")
    return ChartStatus("pending", result.status)


@dataclass(slots=True)
class LiveRomApp:
    action: ActionLabel
    renderer: LiveOverlayRenderer
    required_min_deg: float | None
    history: deque = field(default_factory=lambda: deque(maxlen=180))
    last_result: object = None
    title: str = "ROM"

    def on_frame(self, image, snapshot, rep_index):
        """Main window: camera + skeleton + joint angles + text panel + ROM bar."""
        self.renderer.render(image, snapshot, rep_index)
        draw_progress_bar(image, snapshot.phase.live_rom, self.required_min_deg, y=300)
        return image

    def on_charts(self, canvas, snapshot, rep_index):
        """Chart window: per-frame rolling primary-angle chart."""
        if snapshot.raw_angle is not None:
            self.history.append(snapshot.raw_angle)
        draw_chart(
            canvas, _CHART_RECT, "Primary Angle",
            [ChartSeries("angle", list(self.history), _ANGLE_COLOR)],
            y_bounds=(0.0, _CHART_Y_MAX_DEG),
            status=_chart_status(self.last_result),
        )
        return canvas

    def on_rep(self, snapshot) -> None:
        result = next((r for r in snapshot.rep_results if r.name == "rom"), None)
        if result is not None:
            self.last_result = result
            print(
                f"[rom] status={result.status} passed={result.passed} "
                f"value={result.primary_value} score={result.score}"
            )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    action = (
        ActionLabel.from_value(argv[0]) if argv else ActionLabel.SHOULDER_FLEXION_RIGHT
    )
    references = load_rom_references(settings.paths.rom_reference_path)
    reference = references.get(action)
    app = LiveRomApp(
        action=action,
        renderer=LiveOverlayRenderer(),
        required_min_deg=reference.required_min_deg if reference else None,
    )
    run_live_app(app, settings, build_pose_tracker(settings.pose), RunnerConfig())


if __name__ == "__main__":
    main()

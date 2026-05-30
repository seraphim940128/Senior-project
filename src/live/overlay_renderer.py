"""Live overlay rendering — pure OpenCV drawing, zero metric logic.

Every number drawn here comes straight from a ``PhaseSnapshot``, a
``MetricResult`` or a ``*DebugPayload``: this module holds no state machine,
computes no angle and does no segmentation. The import side of that constraint
is enforced by ``tests/test_no_debug_in_score.py`` (no statistics / numpy /
smoother / private imports). ``cv2`` is imported lazily so the module imports
cleanly without the optional ``vision`` extra.

The chart helpers (``draw_chart`` / ``draw_phase_timeline``) use a debug-view
style: dark axis frame, gridlines, cyan title, bottom legend and a status
badge. Text is drawn light over a black drop-shadow so it stays legible over
the camera image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import CalibrationStatus, JointReadout, LiveCompensation, MetricResult
    from pipeline.phase_tracker import PhaseSnapshot
    from scoring.feedback import Feedback
    from live.session import SessionSnapshot


def _cv2():
    """Lazy ``cv2`` import — keeps this module importable without opencv-python."""
    import cv2

    return cv2


# BGR colors
_WHITE = (245, 245, 245)
_GREY = (150, 150, 150)
_TEXT = (240, 240, 240)        # light foreground for info text
_SHADOW = (0, 0, 0)            # text drop-shadow
_TITLE = (0, 255, 255)         # chart title (cyan)
_AXIS = (80, 80, 80)           # chart axis frame
_GRID = (55, 55, 55)           # chart gridlines
_STATUS_BG = (28, 28, 28)      # chart status badge background
_SKELETON = (120, 230, 120)    # skeleton bones / joints
_GREEN = (60, 170, 60)
_RED = (50, 50, 210)
_AMBER = (20, 150, 200)

# Upper-body skeleton bones.
_UPPER_BODY_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
)
_SKELETON_JOINTS: tuple[str, ...] = tuple(
    sorted({name for pair in _UPPER_BODY_CONNECTIONS for name in pair})
)

_PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (110, 110, 110),
    "ascent": (0, 255, 255),
    "peak": (0, 180, 0),
    "descent": (0, 120, 255),
}
_STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "success": _GREEN,
    "failure": _RED,
    "pending": _AMBER,
}


@dataclass(slots=True)
class OverlayStyle:
    """Layout / typography knobs. The renderer carries no metric state."""

    font_scale: float = 0.72
    thickness: int = 2          # bold fill stroke
    shadow_offset: int = 2      # black drop-shadow offset (px)
    line_height: int = 28
    origin_x: int = 12
    origin_y: int = 32


@dataclass(slots=True)
class ChartSeries:
    """One labelled trace on a chart. ``values`` is built by the caller."""

    label: str
    values: list[float]
    color: tuple[int, int, int]


@dataclass(slots=True)
class ChartStatus:
    """Badge shown at a chart's top-right corner."""

    state: str       # "success" | "failure" | "pending"
    message: str


def draw_text(
    image,
    text: str,
    x: int,
    y: int,
    style: OverlayStyle | None = None,
    color: tuple[int, int, int] | None = None,
    scale: float | None = None,
):
    """Draw light text over a black drop-shadow.

    ``color`` (the foreground) defaults to light grey; ``scale`` overrides the
    style font scale for small chart labels. Public so the demo apps can render
    their own labels with the same look.
    """
    style = style or OverlayStyle()
    cv = _cv2()
    fill = _TEXT if color is None else color
    font_scale = style.font_scale if scale is None else scale
    off = style.shadow_offset
    cv.putText(
        image, text, (x + off, y + off), cv.FONT_HERSHEY_SIMPLEX,
        font_scale, _SHADOW, max(2, style.thickness + 1), cv.LINE_AA,
    )
    cv.putText(
        image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX,
        font_scale, fill, style.thickness, cv.LINE_AA,
    )
    return image


def _put(image, text: str, x: int, y: int, color, style: OverlayStyle) -> None:
    draw_text(image, text, x, y, style, color)


def draw_phase_banner(image, phase: "PhaseSnapshot", style: OverlayStyle | None = None):
    """Phase name + neutral-gate state, read straight from the PhaseSnapshot."""
    style = style or OverlayStyle()
    x, y = style.origin_x, style.origin_y
    _put(image, f"phase: {phase.current_phase}", x, y, _TEXT, style)
    if phase.waiting_for_neutral:
        _put(image, "hold neutral to start", x, y + style.line_height, _AMBER, style)
    elif phase.neutral_ready:
        _put(image, "neutral ready", x, y + style.line_height, _GREEN, style)
    return image


def draw_angle_readout(
    image,
    raw_angle: float | None,
    viz_angle: float | None,
    phase: "PhaseSnapshot",
    style: OverlayStyle | None = None,
):
    """Display angle (EMA viz track) and the live ROM — no recomputation."""
    style = style or OverlayStyle()
    x = style.origin_x
    y = style.origin_y + 2 * style.line_height
    shown = viz_angle if viz_angle is not None else raw_angle
    angle_text = f"{shown:.1f} deg" if shown is not None else "-- deg"
    _put(image, f"angle: {angle_text}", x, y, _TEXT, style)
    if phase.live_rom is not None:
        _put(
            image, f"live ROM: {phase.live_rom:.1f} deg",
            x, y + style.line_height, _TEXT, style,
        )
    return image


def draw_rep_counter(image, rep_index: int, style: OverlayStyle | None = None):
    """Repetition count. ``rep_index`` is owned and supplied by the runner."""
    style = style or OverlayStyle()
    _put(
        image, f"reps: {rep_index}", style.origin_x,
        style.origin_y + 4 * style.line_height, _TEXT, style,
    )
    return image


def draw_metric_summary(
    image,
    rep_results: "list[MetricResult]",
    overall_score: float | None,
    style: OverlayStyle | None = None,
):
    """One line per MetricResult — every value read verbatim from the result."""
    style = style or OverlayStyle()
    x = style.origin_x
    y = style.origin_y + 6 * style.line_height
    score_text = f"{overall_score:.0f}" if overall_score is not None else "--"
    _put(image, f"overall: {score_text}", x, y, _AMBER, style)
    for result in rep_results:
        y += style.line_height
        color = _GREEN if result.passed else _RED if result.passed is False else _TEXT
        value = "--" if result.primary_value is None else f"{result.primary_value:.1f}"
        score = "--" if result.score is None else f"{result.score:.0f}"
        _put(
            image, f"{result.name}: {result.status} val={value} score={score}",
            x, y, color, style,
        )
    return image


def draw_feedback(image, feedback: "Feedback | None", style: OverlayStyle | None = None):
    """Reason codes from the Feedback object — verbatim, no derivation."""
    style = style or OverlayStyle()
    if feedback is None or not feedback.reason_codes:
        return image
    x = style.origin_x
    y = style.origin_y + 14 * style.line_height
    _put(image, "feedback:", x, y, _AMBER, style)
    for code in feedback.reason_codes:
        y += style.line_height
        _put(image, f"- {code}", x, y, _RED, style)
    return image


def draw_progress_bar(
    image,
    current: float | None,
    threshold: float | None,
    y: int,
    style: OverlayStyle | None = None,
    width: int = 260,
    height: int = 18,
):
    """Horizontal bar: ``current`` against a pre-computed ``threshold``.

    Both values arrive ready — this function does no metric lookup.
    """
    style = style or OverlayStyle()
    cv = _cv2()
    x = style.origin_x
    cv.rectangle(image, (x, y), (x + width, y + height), _WHITE, 3)
    cv.rectangle(image, (x, y), (x + width, y + height), _GREY, 1)
    if current is not None and threshold and threshold > 0:
        frac = max(0.0, min(1.0, current / threshold))
        filled = int(frac * width)
        color = _GREEN if current >= threshold else _AMBER
        cv.rectangle(image, (x, y), (x + filled, y + height), color, -1)
    return image


def _draw_chart_status(
    image, anchor: tuple[int, int], status: ChartStatus, style: OverlayStyle
) -> None:
    """Status badge at the chart's top-right corner."""
    cv = _cv2()
    text = f"{status.state.upper()}: {status.message}"
    scale = 0.45
    (text_w, text_h), baseline = cv.getTextSize(
        text, cv.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    pad_x, pad_y = 8, 5
    right, baseline_y = anchor
    left = max(0, right - text_w - pad_x * 2)
    top = max(0, baseline_y - text_h - pad_y)
    bottom = baseline_y + baseline + pad_y
    color = _STATUS_COLORS.get(status.state, _TEXT)
    cv.rectangle(image, (left, top), (right, bottom), _STATUS_BG, -1)
    cv.rectangle(image, (left, top), (right, bottom), color, 1)
    draw_text(image, text, left + pad_x, baseline_y, style, color, scale=scale)


def draw_chart(
    image,
    rect: tuple[int, int, int, int],
    title: str,
    series_list: list[ChartSeries],
    y_bounds: tuple[float, float] | None = None,
    status: ChartStatus | None = None,
    style: OverlayStyle | None = None,
    marker_index: int | None = None,
):
    """Plot one or more pre-built series inside ``rect`` = (x, y, w, h).

    Debug-view style: cyan title, dark axis frame + 3 gridlines, a top-right
    status badge and a bottom legend. Each series is a polyline drawn
    segment-by-segment with ``cv2.line`` (no numpy). The series are built by the
    caller — this function does pure value-to-pixel mapping.

    ``marker_index`` (when set and in range) draws a vertical amber line at that
    sample index — used to mark a window boundary on the trace. The index space
    is the longest series, so the caller must build the marker against that.
    """
    style = style or OverlayStyle()
    cv = _cv2()
    x, y, width, height = rect

    draw_text(image, title, x + 10, y + 24, style, _TITLE, scale=0.6)

    values = [value for series in series_list for value in series.values]
    if status is None:
        status = (
            ChartStatus("success", "streaming") if values
            else ChartStatus("pending", "no data")
        )
    _draw_chart_status(image, (x + width - 12, y + 23), status, style)

    if not values:
        draw_text(image, "No Data", x + 12, y + height // 2, style, _GREY, scale=0.55)
        return image

    if y_bounds is None:
        y_min, y_max = min(values), max(values)
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
    else:
        y_min, y_max = y_bounds
    span = max(y_max - y_min, 1e-6)

    plot_x = x + 12
    plot_y = y + 36
    plot_w = width - 24
    plot_h = height - 54

    # axis frame
    cv.line(image, (plot_x, plot_y), (plot_x + plot_w, plot_y), _AXIS, 1)
    cv.line(image, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h), _AXIS, 1)
    cv.line(image, (plot_x, plot_y), (plot_x, plot_y + plot_h), _AXIS, 1)
    cv.line(image, (plot_x + plot_w, plot_y), (plot_x + plot_w, plot_y + plot_h), _AXIS, 1)
    # 3 internal horizontal gridlines
    for grid_idx in range(1, 4):
        grid_y = int(plot_y + plot_h * grid_idx / 4)
        cv.line(image, (plot_x, grid_y), (plot_x + plot_w, grid_y), _GRID, 1)

    # series polylines
    for series in series_list:
        if len(series.values) < 2:
            continue
        total = len(series.values) - 1
        points = []
        for index, value in enumerate(series.values):
            px = plot_x + int(plot_w * index / max(1, total))
            py = plot_y + plot_h - int(((value - y_min) / span) * plot_h)
            points.append((px, py))
        for start, end in zip(points, points[1:]):
            cv.line(image, start, end, series.color, 2, cv.LINE_AA)

    # optional window-boundary marker (e.g. end of the SE setting-phase window)
    if marker_index is not None:
        marker_total = max((len(s.values) for s in series_list), default=0) - 1
        if marker_total > 0 and 0 <= marker_index <= marker_total:
            mx = plot_x + int(plot_w * marker_index / marker_total)
            cv.line(image, (mx, plot_y), (mx, plot_y + plot_h), _AMBER, 1, cv.LINE_AA)
            draw_text(image, "setting phase", mx + 3, plot_y + 12, style, _AMBER, scale=0.4)

    # bottom legend
    legend_x = x + 12
    legend_y = y + height - 10
    for series in series_list:
        draw_text(image, series.label, legend_x, legend_y, style, series.color, scale=0.45)
        legend_x += max(90, len(series.label) * 11)
    return image


def draw_phase_timeline(
    image,
    rect: tuple[int, int, int, int],
    phases: list[str],
    style: OverlayStyle | None = None,
):
    """Per-frame phase history as a strip of coloured blocks."""
    style = style or OverlayStyle()
    cv = _cv2()
    x, y, width, height = rect
    draw_text(image, "Phase Timeline", x + 10, y + 20, style, _TITLE, scale=0.55)
    if not phases:
        draw_text(image, "No Data", x + 12, y + height // 2, style, _GREY, scale=0.5)
        return image
    usable = max(1, width - 24)
    bar_top = y + 30
    bar_bottom = y + height - 8
    step = usable / max(1, len(phases))
    for index, name in enumerate(phases):
        x1 = int(x + 12 + index * step)
        x2 = int(x + 12 + (index + 1) * step)
        color = _PHASE_COLORS.get(name, _PHASE_COLORS["idle"])
        cv.rectangle(image, (x1, bar_top), (max(x1 + 1, x2), bar_bottom), color, -1)
    return image


def draw_phase_legend(image, origin: tuple[int, int], style: OverlayStyle | None = None):
    """Colour key for the phase timeline."""
    style = style or OverlayStyle()
    cv = _cv2()
    x, y = origin
    for label in ("idle", "ascent", "peak", "descent"):
        cv.rectangle(image, (x, y - 10), (x + 16, y + 6), _PHASE_COLORS[label], -1)
        draw_text(image, label, x + 22, y + 4, style, _TEXT, scale=0.5)
        x += 118
    return image


def draw_skeleton(image, pose_xy, style: OverlayStyle | None = None):
    """Upper-body skeleton from landmark pixel coords.

    ``pose_xy`` maps landmark name -> (x, y) pixels (built by the session).
    Bones are drawn over a dark halo; missing landmarks are skipped.
    """
    style = style or OverlayStyle()
    if not pose_xy:
        return image
    cv = _cv2()
    for name_a, name_b in _UPPER_BODY_CONNECTIONS:
        pa = pose_xy.get(name_a)
        pb = pose_xy.get(name_b)
        if pa is None or pb is None:
            continue
        a = (int(pa[0]), int(pa[1]))
        b = (int(pb[0]), int(pb[1]))
        cv.line(image, a, b, _SHADOW, 5, cv.LINE_AA)
        cv.line(image, a, b, _SKELETON, 2, cv.LINE_AA)
    for name in _SKELETON_JOINTS:
        point = pose_xy.get(name)
        if point is None:
            continue
        center = (int(point[0]), int(point[1]))
        cv.circle(image, center, 5, _SHADOW, -1)
        cv.circle(image, center, 3, _SKELETON, -1)
    return image


def draw_joint_readouts(image, readouts: "list[JointReadout] | None",
                        style: OverlayStyle | None = None):
    """Draw each measured joint's 2D / 3D angle next to the joint.

    ``readouts`` is the session's ``JointReadout`` list; values are drawn
    verbatim (``None`` -> ``--``). 3D is cyan, 2D amber, for quick comparison.
    """
    style = style or OverlayStyle()
    if not readouts:
        return image
    for readout in readouts:
        x = int(readout.x) + 12
        y = int(readout.y)
        a3 = "--" if readout.angle_3d is None else f"{readout.angle_3d:.0f}"
        a2 = "--" if readout.angle_2d is None else f"{readout.angle_2d:.0f}"
        draw_text(image, f"3D {a3}", x, y - 6, style, _TITLE, scale=0.5)
        draw_text(image, f"2D {a2}", x, y + 16, style, _AMBER, scale=0.5)
    return image


def _draw_centered(image, text: str, center_y: int, scale: float, color, style) -> None:
    """Draw ``text`` horizontally centered on the image at vertical ``center_y``."""
    cv = _cv2()
    thickness = max(2, style.thickness + 1)
    (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, (image.shape[1] - text_w) // 2)
    draw_text(image, text, x, center_y + text_h // 2, style, color, scale=scale)


def draw_calibration_banner(
    image, calibration: "CalibrationStatus", style: OverlayStyle | None = None
):
    """Centered resting-posture calibration prompt, drawn before a baseline
    exists. The frame is dimmed so the prompt stands out; for the collecting
    state a progress bar tracks frames captured.
    """
    style = style or OverlayStyle()
    cv = _cv2()
    height, width = image.shape[0], image.shape[1]
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (width, height), _SHADOW, -1)
    cv.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    _draw_centered(image, calibration.message, height // 2, 1.2, _WHITE, style)
    if calibration.state == "collecting" and calibration.frames_target > 0:
        frac = calibration.frames_collected / calibration.frames_target
        frac = max(0.0, min(1.0, frac))
        bar_w = min(width - 80, 420)
        bar_x = (width - bar_w) // 2
        bar_y = height // 2 + 40
        cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 22), _WHITE, 2)
        cv.rectangle(
            image, (bar_x, bar_y), (bar_x + int(bar_w * frac), bar_y + 22), _GREEN, -1
        )
    return image


def draw_live_compensation(
    image,
    live: "LiveCompensation | None",
    style: OverlayStyle | None = None,
    confirmed: bool = False,
    aux_confirmed: bool = False,
):
    """Top-right compensation readout.

    Instantaneous threshold crossings are labelled as possible; confirmed
    labels are supplied only from a completed-repetition evaluator result.
    """
    style = style or OverlayStyle()
    if live is None and not confirmed and not aux_confirmed:
        return image
    cv = _cv2()
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if live is not None and live.trunk_tilt_deviation_deg is not None:
        color = _RED if live.trunk_tilt_over_threshold else _TEXT
        lines.append((f"trunk {live.trunk_tilt_deviation_deg:.1f} deg", color))
    if live is not None and live.shoulder_elevation_ratio is not None:
        color = _RED if live.shoulder_elevation_over_threshold else _TEXT
        lines.append((f"shoulder {live.shoulder_elevation_ratio:+.2f}", color))
    if live is not None and live.aux_shoulder_elevation_ratio is not None:
        color = _AMBER if live.aux_shoulder_elevation_over_threshold else _TEXT
        lines.append((f"other shoulder {live.aux_shoulder_elevation_ratio:+.2f}", color))
    if confirmed:
        lines.insert(0, ("CONFIRMED COMPENSATION", _RED))
    elif live is not None and (
        live.trunk_tilt_over_threshold or live.shoulder_elevation_over_threshold
    ):
        lines.insert(0, ("POSSIBLE COMPENSATION", _RED))
    if aux_confirmed:
        lines.insert(0, ("OTHER SHOULDER OBSERVED", _AMBER))
    elif live is not None and live.aux_shoulder_elevation_over_threshold:
        lines.insert(0, ("POSSIBLE OTHER SHOULDER", _AMBER))
    y = style.origin_y
    for text, color in lines:
        (text_w, _), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
        )
        draw_text(image, text, image.shape[1] - text_w - 12, y, style, color)
        y += style.line_height
    return image


class LiveOverlayRenderer:
    """Assembles the standard overlay panel from a SessionSnapshot.

    Stateless apart from layout ``style``; ``rep_index`` is passed in by the
    runner — the renderer keeps no counter of its own.
    """

    def __init__(self, style: OverlayStyle | None = None) -> None:
        self.style = style or OverlayStyle()

    def render(self, image, snapshot: "SessionSnapshot", rep_index: int):
        draw_skeleton(image, snapshot.pose_xy, self.style)
        calibration = snapshot.calibration
        if calibration is not None and calibration.state in (
            "awaiting", "countdown", "collecting"
        ):
            # No tracking happens during calibration — show only the prompt.
            draw_calibration_banner(image, calibration, self.style)
            return image
        draw_phase_banner(image, snapshot.phase, self.style)
        draw_angle_readout(
            image, snapshot.raw_angle, snapshot.viz_angle, snapshot.phase, self.style
        )
        draw_rep_counter(image, rep_index, self.style)
        if snapshot.rep_results is not None:
            draw_metric_summary(
                image, snapshot.rep_results, snapshot.overall_score, self.style
            )
            draw_feedback(image, snapshot.feedback, self.style)
        draw_joint_readouts(image, snapshot.joint_readouts, self.style)
        compensation = next(
            (
                result
                for result in (snapshot.rep_results or [])
                if result.name == "compensation"
            ),
            None,
        )
        draw_live_compensation(
            image,
            snapshot.live_compensation,
            self.style,
            confirmed=compensation is not None and compensation.passed is False,
            aux_confirmed=(
                compensation is not None
                and compensation.detail.get("aux_shoulder_elevation_detected", 0.0) > 0.0
            ),
        )
        return image

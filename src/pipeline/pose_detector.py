"""MediaPipe pose detection binding.

Pure I/O glue, no metric logic. Supports both the legacy ``mp.solutions.pose``
backend and the newer Tasks ``PoseLandmarker`` backend.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from config import PoseSettings
from models import PoseFrame, TrackedLandmark


POSE_LANDMARK_NAMES = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)


def _normalize_legacy_quality(
    visibility: float | None, presence: float | None
) -> tuple[float | None, float | None]:
    """Legacy ``mp.solutions.pose`` landmarks expose ``visibility`` but no
    ``presence``. :meth:`TrackedLandmark.is_available` requires *both* fields
    to be non-None, so the solutions backend would otherwise reject every
    landmark and drop every frame. Mirror visibility into presence when
    presence is absent; tasks backend supplies both natively and is not
    affected.
    """
    if presence is None and visibility is not None:
        return visibility, visibility
    return visibility, presence


class PoseTracker(Protocol):
    def extract(self, frame: object) -> PoseFrame | None:
        ...

    def describe(self) -> dict[str, object]:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class StubPoseTracker:
    """Returned when pose detection is disabled. Always yields no pose."""

    reason: str = "pose_disabled"

    def extract(self, frame: object) -> PoseFrame | None:
        _ = frame
        return None

    def describe(self) -> dict[str, object]:
        return {"name": "stub_pose_tracker", "mode": "stub", "reason": self.reason}

    def close(self) -> None:
        return None


@dataclass(slots=True)
class MediaPipePoseTracker:
    settings: PoseSettings
    _loaded: bool = field(default=False, init=False)
    _pose: object | None = field(default=None, init=False)
    _cv2: object | None = field(default=None, init=False)
    _mp: object | None = field(default=None, init=False)
    _mp_pose: object | None = field(default=None, init=False)
    _backend: str = field(default="none", init=False)
    _backend_reason: str = field(default="not_initialized", init=False)
    _was_loaded: bool = field(default=False, init=False)

    def extract(self, frame: object) -> PoseFrame | None:
        self._ensure_loaded()
        if not hasattr(frame, "shape"):
            raise TypeError("MediaPipePoseTracker expects numpy.ndarray frames.")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        frame_height, frame_width = rgb.shape[:2]

        if self._backend == "solutions":
            result = self._pose.process(rgb)
            world = getattr(result, "pose_world_landmarks", None)
            if not world:
                return None
            landmark_list = world.landmark
            landmarks_out: dict[str, TrackedLandmark] = {}
            for pose_landmark in self._mp_pose.PoseLandmark:
                idx = int(pose_landmark.value)
                if idx < 0 or idx >= len(landmark_list):
                    continue
                lm = landmark_list[idx]
                vis, pres = _normalize_legacy_quality(
                    getattr(lm, "visibility", None), getattr(lm, "presence", None)
                )
                landmarks_out[pose_landmark.name] = TrackedLandmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=vis, presence=pres,
                )
            display_out: dict[str, TrackedLandmark] | None = None
            image = getattr(result, "pose_landmarks", None)
            if image:
                image_list = image.landmark
                display_out = {}
                for pose_landmark in self._mp_pose.PoseLandmark:
                    idx = int(pose_landmark.value)
                    if idx < 0 or idx >= len(image_list):
                        continue
                    im = image_list[idx]
                    vis, pres = _normalize_legacy_quality(
                        getattr(im, "visibility", None), getattr(im, "presence", None)
                    )
                    display_out[pose_landmark.name] = TrackedLandmark(
                        x=im.x,
                        y=im.y,
                        z=float(getattr(im, "z", 0.0)),
                        visibility=vis,
                        presence=pres,
                    )
            return PoseFrame(
                landmarks=landmarks_out,
                display_landmarks=display_out,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
            )

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._pose.detect(mp_image)
        poses = getattr(result, "pose_world_landmarks", None)
        if not poses or len(poses) == 0:
            return None

        first_pose = poses[0]
        landmarks_out = {}
        for index, landmark in enumerate(first_pose):
            name = POSE_LANDMARK_NAMES[index] if index < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{index}"
            landmarks_out[name] = TrackedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=getattr(landmark, "visibility", None),
                presence=getattr(landmark, "presence", None),
            )
        display_out = None
        image_poses = getattr(result, "pose_landmarks", None)
        if image_poses and len(image_poses) > 0:
            first_image = image_poses[0]
            display_out = {}
            for index, landmark in enumerate(first_image):
                name = POSE_LANDMARK_NAMES[index] if index < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{index}"
                display_out[name] = TrackedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=float(getattr(landmark, "z", 0.0)),
                    visibility=getattr(landmark, "visibility", None),
                    presence=getattr(landmark, "presence", None),
                )
        return PoseFrame(
            landmarks=landmarks_out,
            display_landmarks=display_out,
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )

    def describe(self) -> dict[str, object]:
        backend, reason = (
            (self._backend, self._backend_reason) if self._loaded else self._probe_backend()
        )
        task_model_path = self.settings.task_model_path
        return {
            "name": "mediapipe_pose_tracker",
            "mode": "model",
            "dependency_status": {
                "mediapipe": importlib.util.find_spec("mediapipe") is not None,
                "mediapipe_tasks": importlib.util.find_spec("mediapipe.tasks.python.vision")
                is not None,
            },
            "runtime_ready": backend != "none",
            "backend": backend,
            "backend_reason": reason,
            "backend_preference": self.settings.backend_preference,
            "min_landmark_visibility": self.settings.min_landmark_visibility,
            "min_landmark_presence": self.settings.min_landmark_presence,
            "task_model_path": str(task_model_path) if task_model_path else None,
            "task_model_exists": bool(task_model_path and task_model_path.exists()),
            "loaded": self._loaded,
            "was_loaded": getattr(self, "_was_loaded", self._loaded),
        }

    def close(self) -> None:
        if self._pose is not None and hasattr(self._pose, "close"):
            self._pose.close()
        self._pose = None
        self._was_loaded = self._loaded
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if importlib.util.find_spec("mediapipe") is None:
            raise RuntimeError("mediapipe is unavailable")

        import cv2
        import mediapipe as mp

        backend, reason = self._probe_backend(mp)
        if backend == "none":
            raise RuntimeError(f"MediaPipe pose backend is unavailable: {reason}")

        self._cv2 = cv2
        self._mp = mp
        self._backend = backend
        self._backend_reason = reason

        if backend == "solutions":
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                min_detection_confidence=self.settings.min_detection_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
            )
        else:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_tasks_vision

            options = mp_tasks_vision.PoseLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(
                    model_asset_path=str(self.settings.task_model_path)
                ),
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._pose = mp_tasks_vision.PoseLandmarker.create_from_options(options)
        self._loaded = True

    def _probe_backend(self, mp_module: object | None = None) -> tuple[str, str]:
        if importlib.util.find_spec("mediapipe") is None:
            return "none", "mediapipe_unavailable"

        mp = mp_module
        if mp is None:
            import mediapipe as mp  # type: ignore[no-redef]

        has_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
        has_tasks = importlib.util.find_spec("mediapipe.tasks.python.vision") is not None
        task_model_path = self.settings.task_model_path
        task_model_exists = bool(task_model_path and Path(task_model_path).exists())

        preference = self.settings.backend_preference.strip().lower()
        if preference == "tasks":
            order = ("tasks", "solutions")
        else:
            order = ("solutions", "tasks")

        reasons: list[str] = []
        for backend in order:
            if backend == "solutions":
                if has_solutions:
                    return "solutions", "legacy_solutions"
                reasons.append("solutions_unavailable")
                continue

            if not has_tasks:
                reasons.append("tasks_unavailable")
                continue
            if not task_model_exists:
                reasons.append("task_model_not_found")
                continue
            return "tasks", f"tasks_model:{task_model_path}"

        if not reasons:
            reasons.append("no_pose_backend")
        return "none", ",".join(reasons)


def build_pose_tracker(settings: PoseSettings) -> PoseTracker:
    if not settings.enabled:
        return StubPoseTracker()
    return MediaPipePoseTracker(settings=settings)

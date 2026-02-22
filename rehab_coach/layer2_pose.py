from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import BaselineProfile

Landmark = Tuple[float, float, float]
Landmarks = Dict[str, Landmark]

POSE_LANDMARK_NAMES: List[str] = [
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
]


@dataclass
class SegmentFrame:
    timestamp: float
    confidence: float
    landmarks: Landmarks


def _angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    a_vec = np.array(a, dtype=np.float32)
    b_vec = np.array(b, dtype=np.float32)
    c_vec = np.array(c, dtype=np.float32)

    ba = a_vec - b_vec
    bc = c_vec - b_vec
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _distance_2d(a: Landmark, b: Landmark) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


class PoseExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        task_model_path: Optional[str] = None,
    ):
        self.pose = None
        self.pose_landmarker = None
        self.backend = "none"
        self.backend_reason = "not_initialized"
        self.mp_pose = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.backend = "solutions"
            self.backend_reason = "legacy_solutions"
            return

        if task_model_path:
            model_path = Path(task_model_path)
            if model_path.exists():
                try:
                    from mediapipe.tasks import python as mp_tasks_python
                    from mediapipe.tasks.python import vision as mp_tasks_vision

                    options = mp_tasks_vision.PoseLandmarkerOptions(
                        base_options=mp_tasks_python.BaseOptions(
                            model_asset_path=str(model_path)
                        ),
                        running_mode=mp_tasks_vision.RunningMode.IMAGE,
                        num_poses=1,
                    )
                    self.pose_landmarker = mp_tasks_vision.PoseLandmarker.create_from_options(
                        options
                    )
                    self.backend = "tasks"
                    self.backend_reason = f"tasks_model:{model_path}"
                    return
                except Exception as exc:
                    self.backend = "none"
                    self.backend_reason = f"tasks_init_failed:{exc}"
                    return

        self.backend = "none"
        if task_model_path:
            self.backend_reason = "task_model_not_found"
        else:
            self.backend_reason = "mediapipe_solutions_unavailable"

    def extract(self, frame_bgr: np.ndarray) -> Optional[Landmarks]:
        if self.backend == "solutions" and self.pose is not None and self.mp_pose is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.pose.process(frame_rgb)
            if not result.pose_landmarks:
                return None

            output: Landmarks = {}
            for index, landmark in enumerate(result.pose_landmarks.landmark):
                name = self.mp_pose.PoseLandmark(index).name
                output[name] = (landmark.x, landmark.y, landmark.z)
            return output

        if self.backend == "tasks" and self.pose_landmarker is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.pose_landmarker.detect(mp_image)
            if not result.pose_landmarks:
                return None

            first_pose = result.pose_landmarks[0]
            output: Landmarks = {}
            for index, landmark in enumerate(first_pose):
                if index < len(POSE_LANDMARK_NAMES):
                    name = POSE_LANDMARK_NAMES[index]
                else:
                    name = f"LANDMARK_{index}"
                output[name] = (landmark.x, landmark.y, landmark.z)
            return output

        return None

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()


class Layer2PoseEvaluator:
    def __init__(self, baseline: BaselineProfile):
        self.baseline = baseline

    def _frame_metrics(self, action: str, lm: Landmarks) -> Tuple[float, float, Optional[float], Optional[float]]:
        ls = lm["LEFT_SHOULDER"]
        rs = lm["RIGHT_SHOULDER"]
        le = lm["LEFT_ELBOW"]
        re = lm["RIGHT_ELBOW"]
        lw = lm["LEFT_WRIST"]
        rw = lm["RIGHT_WRIST"]
        lh = lm["LEFT_HIP"]
        rh = lm["RIGHT_HIP"]
        nose = lm["NOSE"]
        lear = lm["LEFT_EAR"]
        rear = lm["RIGHT_EAR"]

        shoulder_width = max(_distance_2d(ls, rs), 1e-6)
        trunk_lateral_lean = abs((rs[1] - ls[1]) / shoulder_width)

        mid_shoulder_x = (ls[0] + rs[0]) / 2.0
        mid_shoulder_y = (ls[1] + rs[1]) / 2.0
        mid_shoulder_z = (ls[2] + rs[2]) / 2.0
        
        mid_hip_x = (lh[0] + rh[0]) / 2.0
        mid_hip_y = (lh[1] + rh[1]) / 2.0
        mid_hip_z = (lh[2] + rh[2]) / 2.0

        trunk_length_2d = float(np.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y))
        trunk_length_2d = max(trunk_length_2d, 1e-6)

        trunk_backward_lean = abs(mid_shoulder_z - mid_hip_z) / trunk_length_2d

        shoulder_elev_l = abs(lh[1] - ls[1])
        shoulder_elev_r = abs(rh[1] - rs[1])

        elbow_flex_l = _angle(ls, le, lw)
        elbow_flex_r = _angle(rs, re, rw)
        shoulder_flex_l = _angle(lh, ls, le)
        shoulder_flex_r = _angle(rh, rs, re)

        if action == "elbow_flexion_left":
            primary = elbow_flex_l
            compensation = max(shoulder_elev_l, trunk_lateral_lean)
            return primary, compensation, None, None

        if action == "elbow_flexion_right":
            primary = elbow_flex_r
            compensation = max(shoulder_elev_r, trunk_lateral_lean)
            return primary, compensation, None, None

        if action == "shoulder_flexion_left":
            primary = shoulder_flex_l
            elbow_drift = abs(180.0 - elbow_flex_l) / 180.0
            compensation = max(trunk_backward_lean, elbow_drift)
            return primary, compensation, None, None

        if action == "shoulder_flexion_right":
            primary = shoulder_flex_r
            elbow_drift = abs(180.0 - elbow_flex_r) / 180.0
            compensation = max(trunk_backward_lean, elbow_drift)
            return primary, compensation, None, None

        if action == "shoulder_abduction_left":
            primary = shoulder_flex_l
            compensation = max(trunk_lateral_lean, shoulder_elev_l)
            return primary, compensation, None, None

        if action == "shoulder_abduction_right":
            primary = shoulder_flex_r
            compensation = max(trunk_lateral_lean, shoulder_elev_r)
            return primary, compensation, None, None

        if action == "shoulder_forward_elevation":
            primary = max(shoulder_flex_l, shoulder_flex_r)
            compensation = max(trunk_backward_lean, shoulder_elev_l, shoulder_elev_r)
            symmetry = abs(shoulder_flex_l - shoulder_flex_r)
            wrist_above = 1.0 if (lw[1] < lear[1] and rw[1] < rear[1]) else 0.0
            return primary, compensation, symmetry, wrist_above

        return 0.0, 0.0, None, None

    def evaluate(self, action: str, segment_id: int, frames: List[SegmentFrame]) -> Optional[dict]:
        if not frames:
            return None

        primary_values: List[float] = []
        comp_values: List[float] = []
        symmetry_values: List[float] = []
        wrist_above_flags: List[float] = []

        for frame in frames:
            primary, comp, symmetry, wrist_above = self._frame_metrics(action, frame.landmarks)
            primary_values.append(primary)
            comp_values.append(comp)
            if symmetry is not None:
                symmetry_values.append(symmetry)
            if wrist_above is not None:
                wrist_above_flags.append(wrist_above)

        primary_peak = max(primary_values) if primary_values else 0.0
        primary_mean = _mean(primary_values)
        
        if len(primary_values) > 1:
            diff_values = [primary_values[i] - primary_values[i-1] for i in range(1, len(primary_values))]
            primary_std = _std(diff_values)
        else:
            primary_std = 0.0

        comp_mean = _mean(comp_values)
        comp_peak = max(comp_values)

        symmetry_peak_diff = max(symmetry_values) if symmetry_values else None
        wrist_above_ratio = _mean(wrist_above_flags) if wrist_above_flags else None

        rom_max = self.baseline.rom_max.get(action, 170.0)
        reach_ratio = primary_peak / max(rom_max, 1e-6)

        comp_base = self.baseline.comp_base.get(action, 0.2)
        comp_std = self.baseline.comp_std.get(action, 0.08)
        stab_base = self.baseline.stability_base.get(action, 8.0)
        stab_std = self.baseline.stability_std.get(action, 3.0)

        if reach_ratio >= 0.85:
            primary_status = "good"
        elif reach_ratio >= 0.75:
            primary_status = "acceptable"
        else:
            primary_status = "insufficient"

        if comp_peak <= comp_base + comp_std:
            comp_status = "none"
        elif comp_peak <= comp_base + 2.0 * comp_std:
            comp_status = "mild"
        else:
            comp_status = "excessive"

        if symmetry_peak_diff is None:
            symmetry_status = "not_applicable"
        else:
            sym_base = self.baseline.symmetry_base.get(action, 10.0)
            sym_std = self.baseline.symmetry_std.get(action, 5.0)
            if symmetry_peak_diff <= sym_base + 2.0 * sym_std:
                symmetry_status = "balanced"
            else:
                symmetry_status = "imbalanced"

        stability_threshold = stab_base + 2.0 * stab_std
        stability_status = "stable" if primary_std <= stability_threshold else "unstable"

        segment_duration = max(0.0, frames[-1].timestamp - frames[0].timestamp)
        confidence_mean = _mean([item.confidence for item in frames])

        return {
            "action": action,
            "segment": {
                "id": segment_id,
                "duration_s": round(segment_duration, 3),
                "confidence_mean": round(confidence_mean, 4),
            },
            "metrics": {
                "primary_peak": round(primary_peak, 3),
                "primary_mean": round(primary_mean, 3),
                "primary_std": round(primary_std, 3),
                "reach_ratio": round(reach_ratio, 3),
                "comp_peak": round(comp_peak, 3),
                "comp_mean": round(comp_mean, 3),
                "symmetry_peak_diff": round(symmetry_peak_diff, 3) if symmetry_peak_diff is not None else None,
                "wrist_above_head_ratio": round(wrist_above_ratio, 3) if wrist_above_ratio is not None else None,
            },
            "posture_summary": {
                "primary_joint_range": primary_status,
                "compensation": comp_status,
                "symmetry": symmetry_status,
                "stability": stability_status,
            },
        }

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

def _calculate_kinematics_derivatives(positions: List[np.ndarray], timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """計算給定 3D 軌跡的速度與加速度大小，並回傳平均採樣頻率"""
    if len(positions) < 3:
        return np.array([]), np.array([]), 30.0
        
    pos_arr = np.array(positions)
    t_arr = np.array(timestamps)
    
    # 計算時間差與平均採樣率
    dt = np.diff(t_arr)
    dt[dt <= 0] = 1e-3 
    fs = 1.0 / np.mean(dt)
    
    # 一階微分：速度
    dp = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
    velocity = dp / dt
    
    # 二階微分：加速度
    dv = np.diff(velocity)
    dt_a = dt[1:]
    acceleration = dv / dt_a
    
    return velocity, acceleration, fs

def _calculate_sparc(velocity_profile: np.ndarray, fs: float, fc: float = 10.0) -> float:
    """
    計算 SPARC (Spectral Arc Length) 平滑度指標
    基於速度訊號的傅立葉幅度頻譜，數值越接近 0 代表越平滑。
    """
    if len(velocity_profile) < 5:
        return 0.0
        
    # 進行快速傅立葉轉換 (FFT)
    nfft = int(pow(2, np.ceil(np.log2(len(velocity_profile))) + 4))
    V = np.fft.rfft(velocity_profile, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    
    # 頻譜幅度正規化
    V_mag = np.abs(V)
    V_mag_norm = V_mag / (max(V_mag[0], 1e-6))
    
    # 擷取目標低頻帶 (0 到 fc)
    indices = np.where(freqs <= fc)[0]
    if len(indices) < 2:
        return 0.0
        
    V_band = V_mag_norm[indices]
    f_band = freqs[indices]
    
    # 計算頻譜曲線之弧長
    df = f_band[1] - f_band[0]
    dV_df = np.diff(V_band) / df
    
    # SPARC 是一個無因次的負值指標
    arc_length = np.sum(np.sqrt((1.0/fc)**2 + dV_df**2)) * df
    return float(-arc_length)

def _calculate_rms(acceleration_profile: np.ndarray) -> float:
    """計算 RMS 加速度，衡量微小內部震顫與動態穩定度"""
    if len(acceleration_profile) == 0:
        return 0.0
    return float(np.sqrt(np.mean(acceleration_profile**2)))

def _vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """計算兩空間向量之夾角 (度)"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


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

    def _extract_frame_kinematics(self, action: str, lm: Landmarks) -> dict:
        """提取單幀的運動學空間向量與特徵參數"""
        ls, rs = np.array(lm["LEFT_SHOULDER"]), np.array(lm["RIGHT_SHOULDER"])
        le, re = np.array(lm["LEFT_ELBOW"]), np.array(lm["RIGHT_ELBOW"])
        lw, rw = np.array(lm["LEFT_WRIST"]), np.array(lm["RIGHT_WRIST"])
        lh, rh = np.array(lm["LEFT_HIP"]), np.array(lm["RIGHT_HIP"])

        # 定義 MediaPipe 空間中的垂直向上向量
        up_vec = np.array([0.0, -1.0, 0.0])

        # 1. 軀幹代償計算 (Trunk Compensation)
        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0
        tr_vec = mid_shoulder - mid_hip
        
        # 矢狀面前傾/後傾 (Y-Z 平面投影)
        trunk_sagittal = _vector_angle(np.array([0.0, tr_vec[1], tr_vec[2]]), up_vec)
        # 額狀面側彎 (X-Y 平面投影)
        trunk_frontal = _vector_angle(np.array([tr_vec[0], tr_vec[1], 0.0]), up_vec)
        trunk_comp_angle = max(trunk_sagittal, trunk_frontal)

        # 2. 肢體長度 (用於後續聳肩位移標準化)
        arm_len_l = max(np.linalg.norm(le - ls) + np.linalg.norm(lw - le), 1e-6)
        arm_len_r = max(np.linalg.norm(re - rs) + np.linalg.norm(rw - re), 1e-6)

        # 3. 典型關節角度 (Primary ROM) 計算
        primary_angle = 0.0
        
        if "elbow_flexion_left" in action:
            primary_angle = 180.0 - _vector_angle(le - ls, lw - le)
        elif "elbow_flexion_right" in action:
            primary_angle = 180.0 - _vector_angle(re - rs, rw - re)
        elif "shoulder_flexion_left" in action:
            se_l_yz = np.array([0.0, le[1] - ls[1], le[2] - ls[2]])
            primary_angle = _vector_angle(se_l_yz, up_vec)
        elif "shoulder_flexion_right" in action:
            se_r_yz = np.array([0.0, re[1] - rs[1], re[2] - rs[2]])
            primary_angle = _vector_angle(se_r_yz, up_vec)
        elif "shoulder_abduction_left" in action:
            se_l_xy = np.array([le[0] - ls[0], le[1] - ls[1], 0.0])
            primary_angle = _vector_angle(se_l_xy, up_vec)
        elif "shoulder_abduction_right" in action:
            se_r_xy = np.array([re[0] - rs[0], re[1] - rs[1], 0.0])
            primary_angle = _vector_angle(se_r_xy, up_vec)
        elif "shoulder_forward_elevation" in action:
            scaption_l = _vector_angle(le - ls, up_vec)
            scaption_r = _vector_angle(re - rs, up_vec)
            primary_angle = max(scaption_l, scaption_r)

        return {
            "primary": primary_angle,
            "trunk_comp": trunk_comp_angle,
            "shoulder_y_l": ls[1],
            "shoulder_y_r": rs[1],
            "arm_len_l": arm_len_l,
            "arm_len_r": arm_len_r
        }

    def evaluate(self, action: str, segment_id: int, frames: List[SegmentFrame]) -> Optional[dict]:
        if not frames:
            return None

        kinematics_seq = [self._extract_frame_kinematics(action, f.landmarks) for f in frames]
        
        primary_values = [k["primary"] for k in kinematics_seq]
        trunk_comp_values = [k["trunk_comp"] for k in kinematics_seq]
        
        initial_trunk = np.mean(trunk_comp_values[:3]) if len(trunk_comp_values) >= 3 else trunk_comp_values[0]
        relative_trunk_comp = [abs(val - initial_trunk) for val in trunk_comp_values]

        primary_peak = max(primary_values)
        primary_mean = _mean(primary_values)
        primary_std = _std([primary_values[i] - primary_values[i-1] for i in range(1, len(primary_values))]) if len(primary_values) > 1 else 0.0
        
        comp_peak = max(relative_trunk_comp) if relative_trunk_comp else 0.0
        comp_mean = _mean(relative_trunk_comp)

        rom_max = self.baseline.rom_max.get(action, 135.0)
        reach_ratio = primary_peak / max(rom_max, 1e-6)

        comp_base = self.baseline.comp_base.get(action, 5.0)
        comp_std = self.baseline.comp_std.get(action, 2.5)

        primary_status = "good" if reach_ratio >= 0.90 else ("acceptable" if reach_ratio >= 0.75 else "insufficient")
        
        if comp_peak <= comp_base + comp_std:
            comp_status = "none"
        elif comp_peak <= comp_base + 3.0 * comp_std:
            comp_status = "mild"
        else:
            comp_status = "excessive"

        # ======== 計算 SPARC 與 RMS ========
        target_joint_positions = []
        for f in frames:
            if "left" in action:
                target_joint_positions.append(np.array(f.landmarks["LEFT_WRIST"]))
            else:
                target_joint_positions.append(np.array(f.landmarks["RIGHT_WRIST"]))
        
        timestamps = [f.timestamp for f in frames]
        
        velocity, acceleration, fs = _calculate_kinematics_derivatives(target_joint_positions, timestamps)
        
        sparc_val = _calculate_sparc(velocity, fs, fc=10.0)
        rms_acc = _calculate_rms(acceleration)

        stability_status = "stable" if (sparc_val >= -6.0 and rms_acc <= 8.0) else "unstable"

        segment_duration = max(0.0, frames[-1].timestamp - frames[0].timestamp)

        return {
            "action": action,
            "segment": {
                "id": segment_id,
                "duration_s": round(segment_duration, 3),
                "confidence_mean": round(_mean([item.confidence for item in frames]), 4),
            },
            "metrics": {
                "primary_peak": round(primary_peak, 3),
                "primary_mean": round(primary_mean, 3),
                "primary_std": round(primary_std, 3),
                "reach_ratio": round(reach_ratio, 3),
                "comp_peak": round(comp_peak, 3),
                "comp_mean": round(comp_mean, 3),
                "sparc": round(sparc_val, 3),
                "rms_acc": round(rms_acc, 3),
            },
            "posture_summary": {
                "primary_joint_range": primary_status,
                "compensation": comp_status,
                "symmetry": "not_applicable",
                "stability": stability_status,
            },
        }
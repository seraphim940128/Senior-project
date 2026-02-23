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
class LandmarkFilter:
    """用於平滑 MediaPipe 輸出座標的指數平滑濾波器"""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.filtered_landmarks: Optional[Landmarks] = None

    def update(self, current_landmarks: Landmarks) -> Landmarks:
        if self.filtered_landmarks is None:
            self.filtered_landmarks = current_landmarks.copy()
            return self.filtered_landmarks

        for key in current_landmarks:
            if key in self.filtered_landmarks:
                curr_x, curr_y, curr_z = current_landmarks[key]
                prev_x, prev_y, prev_z = self.filtered_landmarks[key]
                new_x = prev_x + self.alpha * (curr_x - prev_x)
                new_y = prev_y + self.alpha * (curr_y - prev_y)
                new_z = prev_z + self.alpha * (curr_z - prev_z)
                self.filtered_landmarks[key] = (new_x, new_y, new_z)
        return self.filtered_landmarks.copy()
    
def _calculate_kinematics_derivatives(positions: List[np.ndarray], timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """計算給定 3D 軌跡的速度與加速度大小，並回傳平均採樣頻率"""
    if len(positions) < 3:
        return np.array([]), np.array([]), 30.0
        
    pos_arr = np.array(positions)
    t_arr = np.array(timestamps)
    
    dt = np.diff(t_arr)
    dt[dt <= 0] = 1e-3 
    fs = 1.0 / np.mean(dt)
    
    dp = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
    velocity = dp / dt
    
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
        
    nfft = int(pow(2, np.ceil(np.log2(len(velocity_profile))) + 4))
    V = np.fft.rfft(velocity_profile, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    
    V_mag = np.abs(V)
    V_mag_norm = V_mag / (max(V_mag[0], 1e-6))
    
    indices = np.where(freqs <= fc)[0]
    if len(indices) < 2:
        return 0.0
        
    V_band = V_mag_norm[indices]
    f_band = freqs[indices]
    
    df = f_band[1] - f_band[0]
    dV_df = np.diff(V_band) / df
    
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

        up_vec = np.array([0.0, -1.0, 0.0])

        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0
        tr_vec = mid_shoulder - mid_hip
        
        z_penalty = 0.3
        tr_vec_penalized = np.array([tr_vec[0], tr_vec[1], tr_vec[2] * z_penalty])

        trunk_sagittal = _vector_angle(np.array([0.0, tr_vec_penalized[1], tr_vec_penalized[2]]), up_vec)
        trunk_frontal = _vector_angle(np.array([tr_vec_penalized[0], tr_vec_penalized[1], 0.0]), up_vec)

        arm_len_l = max(np.linalg.norm(le - ls) + np.linalg.norm(lw - le), 1e-6)
        arm_len_r = max(np.linalg.norm(re - rs) + np.linalg.norm(rw - re), 1e-6)

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
            shoulder_flex_l = _vector_angle(ls - lh, le - ls)
            shoulder_flex_r = _vector_angle(rs - rh, re - rs)
            primary_angle = max(shoulder_flex_l, shoulder_flex_r)

        return {
            "primary": primary_angle,
            "trunk_sagittal": trunk_sagittal,
            "trunk_frontal": trunk_frontal,
            "sh_to_hip_l": lh[1] - ls[1],
            "sh_to_hip_r": rh[1] - rs[1],
            "arm_len_l": arm_len_l,
            "arm_len_r": arm_len_r
        }

    def evaluate(self, action: str, segment_id: int, frames: List[SegmentFrame]) -> Optional[dict]:
        if not frames:
            return None

        lm_filter = LandmarkFilter(alpha=0.5)
        filtered_kinematics_seq = []
        for f in frames:
            smoothed_lm = lm_filter.update(f.landmarks)
            filtered_kinematics_seq.append(self._extract_frame_kinematics(action, smoothed_lm))
        
        kinematics_seq = filtered_kinematics_seq
        
        # 關節活動度 (ROM)
        primary_values = [k["primary"] for k in kinematics_seq]
        primary_peak = max(primary_values)
        primary_mean = _mean(primary_values)
        primary_std = _std([primary_values[i] - primary_values[i-1] for i in range(1, len(primary_values))]) if len(primary_values) > 1 else 0.0
        
        rom_max = self.baseline.rom_max.get(action, 135.0)
        reach_ratio = primary_peak / max(rom_max, 1e-6)
        primary_status = "good" if reach_ratio >= 0.90 else ("acceptable" if reach_ratio >= 0.75 else "insufficient")

        # 代償行為量化
        comp_issues = []

        init_sagittal = np.mean([k["trunk_sagittal"] for k in kinematics_seq[:3]]) if len(kinematics_seq) >= 3 else kinematics_seq[0]["trunk_sagittal"]
        init_frontal = np.mean([k["trunk_frontal"] for k in kinematics_seq[:3]]) if len(kinematics_seq) >= 3 else kinematics_seq[0]["trunk_frontal"]
        
        max_sagittal_dev = max([abs(k["trunk_sagittal"] - init_sagittal) for k in kinematics_seq])
        max_frontal_dev = max([abs(k["trunk_frontal"] - init_frontal) for k in kinematics_seq])
        
        if "elbow" in action:
            sagittal_thresh, frontal_thresh = 10.0, 10.0
        elif "forward_elevation" in action:
            sagittal_thresh, frontal_thresh = 25.0, 15.0 
        else: 
            sagittal_thresh, frontal_thresh = 20.0, 15.0
            
        if max_sagittal_dev > sagittal_thresh or max_frontal_dev > frontal_thresh:
            comp_issues.append("trunk_lean")

        is_left_action = "left" in action or "elevation" in action
        is_right_action = "right" in action or "elevation" in action
        
        hiking_detected = False
        
        dynamic_hiking_thresh = 0.08 + 0.10 * (min(primary_peak, 180.0) / 180.0)

        if is_left_action:
            init_sh_dist_l = np.mean([k["sh_to_hip_l"] for k in kinematics_seq[:3]])
            max_sh_dist_l = max([k["sh_to_hip_l"] for k in kinematics_seq])
            arm_len_l = kinematics_seq[0]["arm_len_l"]
            sv_norm_l = max(0.0, (max_sh_dist_l - init_sh_dist_l)) / arm_len_l
            if sv_norm_l > dynamic_hiking_thresh:
                hiking_detected = True

        if is_right_action:
            init_sh_dist_r = np.mean([k["sh_to_hip_r"] for k in kinematics_seq[:3]])
            max_sh_dist_r = max([k["sh_to_hip_r"] for k in kinematics_seq])
            arm_len_r = kinematics_seq[0]["arm_len_r"]
            sv_norm_r = max(0.0, (max_sh_dist_r - init_sh_dist_r)) / arm_len_r
            if sv_norm_r > dynamic_hiking_thresh:
                hiking_detected = True

        if hiking_detected:
            comp_issues.append("shoulder_hiking")

        if not comp_issues:
            comp_status = "none"
        else:
            comp_status = ",".join(comp_issues)

        # 計算 SPARC 與 RMS
        target_joint_positions = []
        for f in frames:
            if "left" in action:
                target_joint_positions.append(np.array(f.landmarks["LEFT_WRIST"]))
            elif "right" in action:
                target_joint_positions.append(np.array(f.landmarks["RIGHT_WRIST"]))
            else:
                target_joint_positions.append((np.array(f.landmarks["LEFT_WRIST"]) + np.array(f.landmarks["RIGHT_WRIST"])) / 2.0)
        
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
                "trunk_dev_max": round(max(max_sagittal_dev, max_frontal_dev), 3),
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
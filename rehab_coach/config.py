from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


MODEL_LABELS: List[str] = [
    "shoulder_abduction_left",
    "shoulder_abduction_right",
    "shoulder_flexion_left",
    "shoulder_flexion_right",
    "side_tap_left",
    "side_tap_right",
    "elbow_flexion_left",
    "elbow_flexion_right",
    "shoulder_forward_elevation",
]

SUPPORTED_ACTIONS: List[str] = [
    "elbow_flexion_left",
    "elbow_flexion_right",
    "shoulder_flexion_left",
    "shoulder_flexion_right",
    "shoulder_abduction_left",
    "shoulder_abduction_right",
    "shoulder_forward_elevation",
]

UNITY_ACTION_COMMAND_MAP: Dict[str, str] = {
    "shoulder_abduction_left": "u",
    "shoulder_abduction_right": "o",
    "shoulder_flexion_left": "j",
    "shoulder_flexion_right": "l",
    "shoulder_forward_elevation": "i",
}


def _main_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class ModelPaths:
    main_dir: Path = field(default_factory=_main_dir)
    action_weights: Path = field(init=False)
    yolo_weights: Path = field(init=False)

    def __post_init__(self) -> None:
        self.action_weights = self.main_dir / "best_transformer_model_25_topk.pt"
        self.yolo_weights = self.main_dir / "yolo11n.pt"


@dataclass
class ActionConfig:
    sequence_len: int = 30
    stable_frames: int = 3
    confidence_threshold: float = 0.65
    cooldown_s: float = 1.8


@dataclass
class BaselineProfile:
    rom_max: Dict[str, float] = field(
        default_factory=lambda: {
            "elbow_flexion_left": 145.0,
            "elbow_flexion_right": 145.0,
            "shoulder_flexion_left": 170.0,
            "shoulder_flexion_right": 170.0,
            "shoulder_abduction_left": 170.0,
            "shoulder_abduction_right": 170.0,
            "shoulder_forward_elevation": 170.0,
        }
    )
    comp_base: Dict[str, float] = field(
        default_factory=lambda: {
            "elbow_flexion_left": 0.25,
            "elbow_flexion_right": 0.25,
            "shoulder_flexion_left": 0.15,
            "shoulder_flexion_right": 0.15,
            "shoulder_abduction_left": 0.25,
            "shoulder_abduction_right": 0.25,
            "shoulder_forward_elevation": 0.20,
        }
    )
    comp_std: Dict[str, float] = field(
        default_factory=lambda: {
            "elbow_flexion_left": 0.08,
            "elbow_flexion_right": 0.08,
            "shoulder_flexion_left": 0.06,
            "shoulder_flexion_right": 0.06,
            "shoulder_abduction_left": 0.08,
            "shoulder_abduction_right": 0.08,
            "shoulder_forward_elevation": 0.08,
        }
    )
    stability_base: Dict[str, float] = field(
        default_factory=lambda: {
            "elbow_flexion_left": 6.0,
            "elbow_flexion_right": 6.0,
            "shoulder_flexion_left": 8.0,
            "shoulder_flexion_right": 8.0,
            "shoulder_abduction_left": 8.0,
            "shoulder_abduction_right": 8.0,
            "shoulder_forward_elevation": 8.0,
        }
    )
    stability_std: Dict[str, float] = field(
        default_factory=lambda: {
            "elbow_flexion_left": 3.0,
            "elbow_flexion_right": 3.0,
            "shoulder_flexion_left": 3.0,
            "shoulder_flexion_right": 3.0,
            "shoulder_abduction_left": 3.0,
            "shoulder_abduction_right": 3.0,
            "shoulder_forward_elevation": 3.0,
        }
    )
    symmetry_base: Dict[str, float] = field(
        default_factory=lambda: {"shoulder_forward_elevation": 10.0}
    )
    symmetry_std: Dict[str, float] = field(
        default_factory=lambda: {"shoulder_forward_elevation": 5.0}
    )


@dataclass
class AppConfig:
    model_labels: List[str] = field(default_factory=lambda: MODEL_LABELS.copy())
    supported_actions: List[str] = field(default_factory=lambda: SUPPORTED_ACTIONS.copy())
    models: ModelPaths = field(default_factory=ModelPaths)
    action: ActionConfig = field(default_factory=ActionConfig)
    baseline: BaselineProfile = field(default_factory=BaselineProfile)
    unity_host: str = "127.0.0.1"
    unity_port: int = 5500
    unity_action_map: Dict[str, str] = field(
        default_factory=lambda: UNITY_ACTION_COMMAND_MAP.copy()
    )
    unity_confidence_threshold: float = 0.90
    unity_command_cooldown_s: float = 3.0

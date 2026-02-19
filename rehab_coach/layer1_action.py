from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import roi_align
from ultralytics import YOLO

from .config import AppConfig
from .models import OnlineFeatureBuffer, load_classifier


def letterbox_224(rgb_frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    height, width = rgb_frame.shape[:2]
    scale = 224.0 / max(height, width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))

    resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    top = (224 - new_h) // 2
    bottom = 224 - new_h - top
    left = (224 - new_w) // 2
    right = 224 - new_w - left
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, left, top


@dataclass
class ActionPrediction:
    action_label: str
    confidence: float
    is_stable: bool
    segment_id: Optional[int]
    is_supported: bool
    stable_count: int
    stable_frames: int


class Layer1ActionRecognizer:
    def __init__(self, config: AppConfig, device: torch.device):
        self.config = config
        self.device = device

        self.yolo = YOLO(str(config.models.yolo_weights))
        self.yolo.to("cuda:0" if device.type == "cuda" else "cpu")
        self.yolo_device = 0 if device.type == "cuda" else "cpu"

        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features.to(device).eval()

        self.transform_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.classifier = load_classifier(
            str(config.models.action_weights),
            num_classes=len(config.model_labels),
            device=device,
        )
        self.buffer = OnlineFeatureBuffer(sequence_len=config.action.sequence_len)

        self._last_label: Optional[str] = None
        self._stable_count = 0
        self._in_stable = False
        self._stable_label: Optional[str] = None
        self._segment_id = 0

    def _extract_feature(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_224, _, _, _ = letterbox_224(rgb)

        results = self.yolo.predict(
            frame_224,
            classes=[0],
            conf=0.25,
            verbose=False,
            device=self.yolo_device,
        )
        if not results:
            return None

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None or boxes.xyxy.shape[0] == 0:
            return None

        confidences = getattr(boxes, "conf", None)
        index = int(torch.argmax(confidences).item()) if confidences is not None else 0
        x1, y1, x2, y2 = boxes.xyxy[index].detach().cpu().numpy().tolist()

        image_tensor = self.transform_norm(frame_224).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fmap = self.feature_extractor(image_tensor)

        spatial_scale = fmap.shape[-1] / 224.0
        boxes_224 = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        pooled = roi_align(
            fmap,
            [boxes_224],
            output_size=(3, 3),
            spatial_scale=spatial_scale,
            sampling_ratio=-1,
            aligned=True,
        )
        return pooled.squeeze(0)

    def _update_stability(self, label: str, confidence: float) -> Tuple[bool, Optional[int]]:
        threshold = self.config.action.confidence_threshold
        if confidence >= threshold and label == self._last_label:
            self._stable_count += 1
        elif confidence >= threshold:
            self._stable_count = 1
        else:
            self._stable_count = 0

        self._last_label = label
        is_supported = label in self.config.supported_actions
        is_stable = self._stable_count >= self.config.action.stable_frames and is_supported

        if is_stable:
            if not self._in_stable or label != self._stable_label:
                self._segment_id += 1
            self._in_stable = True
            self._stable_label = label
            return True, self._segment_id

        self._in_stable = False
        self._stable_label = None
        return False, None

    def predict(self, frame: np.ndarray) -> Optional[ActionPrediction]:
        feature = self._extract_feature(frame)
        if feature is None:
            if not self.buffer.ready():
                return None
            feature = self.buffer.last()

        self.buffer.push(feature)
        sequence = self.buffer.tensor(self.device)

        with torch.no_grad():
            logits = self.classifier(sequence)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        label = self.config.model_labels[pred_idx]
        confidence = float(probs[pred_idx])
        is_supported = label in self.config.supported_actions
        is_stable, segment_id = self._update_stability(label, confidence)

        return ActionPrediction(
            action_label=label,
            confidence=confidence,
            is_stable=is_stable,
            segment_id=segment_id,
            is_supported=is_supported,
            stable_count=self._stable_count,
            stable_frames=self.config.action.stable_frames
        )

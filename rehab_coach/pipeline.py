from __future__ import annotations

from typing import List, Optional

import numpy as np

from .layer1_action import ActionPrediction, Layer1ActionRecognizer
from .layer2_pose import Layer2PoseEvaluator, PoseExtractor, SegmentFrame


class RehabCoachPipeline:
    """整合 Layer1 與 Layer2，輸出片段摘要與回饋事件。"""

    def __init__(
        self,
        layer1: Layer1ActionRecognizer,
        pose_extractor: PoseExtractor,
        layer2: Layer2PoseEvaluator,
        cooldown_s: float = 1.8,
    ):
        self.layer1 = layer1
        self.pose_extractor = pose_extractor
        self.layer2 = layer2
        self.cooldown_s = cooldown_s

        self.active_segment_id: Optional[int] = None
        self.active_action: Optional[str] = None
        self.segment_frames: List[SegmentFrame] = []
        self.last_feedback_ts: float = 0.0

    def _needs_feedback(self, summary: dict) -> bool:
        posture = summary.get("posture_summary", {})
        return (
            posture.get("primary_joint_range") == "insufficient"
            or posture.get("compensation") == "excessive"
            or posture.get("symmetry") == "imbalanced"
            or posture.get("stability") == "unstable"
        )

    def _finalize_segment(self) -> Optional[dict]:
        if not self.segment_frames or self.active_segment_id is None or self.active_action is None:
            self.segment_frames = []
            self.active_segment_id = None
            self.active_action = None
            return None

        summary = self.layer2.evaluate(
            action=self.active_action,
            segment_id=self.active_segment_id,
            frames=self.segment_frames,
        )

        self.segment_frames = []
        self.active_segment_id = None
        self.active_action = None
        return summary

    def process_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        prediction: Optional[ActionPrediction] = self.layer1.predict(frame)
        landmarks = self.pose_extractor.extract(frame)

        summary = None
        feedback_event = None

        in_stable_segment = (
            prediction is not None
            and prediction.is_stable
            and prediction.segment_id is not None
            and landmarks is not None
        )

        if in_stable_segment:
            if (
                self.active_segment_id is not None
                and (
                    prediction.segment_id != self.active_segment_id
                    or prediction.action_label != self.active_action
                )
            ):
                summary = self._finalize_segment()

            if self.active_segment_id is None:
                self.active_segment_id = prediction.segment_id
                self.active_action = prediction.action_label

            self.segment_frames.append(
                SegmentFrame(
                    timestamp=timestamp,
                    confidence=prediction.confidence,
                    landmarks=landmarks,
                )
            )
        elif self.active_segment_id is not None:
            summary = self._finalize_segment()

        if summary is not None and self._needs_feedback(summary):
            if timestamp - self.last_feedback_ts >= self.cooldown_s:
                feedback_event = summary
                self.last_feedback_ts = timestamp

        return {
            "prediction": prediction,
            "summary": summary,
            "feedback_event": feedback_event,
        }

    def close(self) -> None:
        self.pose_extractor.close()

# export OPENAI_API_KEY="sk-......"

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


ACTION_ZH = {
    "elbow_flexion_left": "左肘屈曲",
    "elbow_flexion_right": "右肘屈曲",
    "shoulder_flexion_left": "左肩屈曲",
    "shoulder_flexion_right": "右肩屈曲",
    "shoulder_abduction_left": "左肩外展",
    "shoulder_abduction_right": "右肩外展",
    "shoulder_forward_elevation": "雙手前舉過頭",
}


@dataclass
class Layer3FeedbackGenerator:
    use_llm: bool = False
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"

    def generate(self, summary: Dict) -> Dict[str, str]:
        if self.use_llm:
            llm_result = self._generate_with_llm(summary)
            if llm_result is not None:
                return llm_result
        return self._generate_template(summary)

    def _generate_template(self, summary: Dict) -> Dict[str, str]:
        action = summary.get("action", "")
        action_zh = ACTION_ZH.get(action, action)

        posture = summary.get("posture_summary", {})
        issues = []
        fixes = []

        if posture.get("primary_joint_range") == "insufficient":
            issues.append("動作幅度還不夠")
            fixes.append("再抬高一點，慢慢到可控制的最高點")
        if posture.get("compensation") == "excessive":
            issues.append("有明顯代償")
            fixes.append("先放鬆肩膀，軀幹保持正中")
        if posture.get("symmetry") == "imbalanced":
            issues.append("左右不夠對稱")
            fixes.append("兩側速度和高度盡量一致")
        if posture.get("sta a  bility") == "unstable":
            issues.append("動作不夠穩定")
            fixes.append("節奏放慢，維持平順出力")

        if not issues:
            coach_text = f"{action_zh}做得很好，維持現在的節奏和姿勢。"
            ui_hint = "動作穩定，繼續保持"
            return {"coach_text": coach_text, "ui_hint": ui_hint}

        issue_text = "、".join(issues[:2])
        fix_text = "；".join(fixes[:2])
        coach_text = f"{action_zh}目前{issue_text}，{fix_text}。你做得不錯，繼續加油。"
        ui_hint = fixes[0] if fixes else "保持穩定"
        return {"coach_text": coach_text, "ui_hint": ui_hint}

    def _generate_with_llm(self, summary: Dict) -> Optional[Dict[str, str]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        action = summary.get("action", "")
        action_zh = ACTION_ZH.get(action, action)
        posture = summary.get("posture_summary", {})
        metrics = summary.get("metrics", {})

        system_prompt = (
            "你是一位專業、親切且富有同理心的復健物理治療師。"
            "你的任務是根據傳入的動作數據，給出給患者一句口語化的即時反饋。"
            "規則："
            "1. 語氣要鼓勵，不要嚴厲責備。"
            "2. 針對 'metrics' 中的具體數值（如角度）進行模糊化指導（例如：'手臂再抬高一點' 而不是 '目前只有 85 度'）。"
            "3. 如果 'posture_summary' 中有問題，優先指出最嚴重的一個。"
            "4. 輸出必須是繁體中文。"
            "5. 回傳格式必須嚴格為 JSON：{\"coach_text\": \"語音播放用的長句\", \"ui_hint\": \"螢幕顯示的短句(5字內)\"}。"
        )
        user_prompt = (
            f"動作名稱：{action_zh}\n"
            f"姿態分析摘要：{posture}\n"
            f"詳細數據指標：{metrics}\n"
            "請生成教練反饋 JSON。"
        )

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_output_tokens=120,
            )
            text = (getattr(response, "output_text", None) or "").strip()
            if not text:
                return None

            import json

            payload = json.loads(text)
            coach_text = str(payload.get("coach_text", "")).strip()
            ui_hint = str(payload.get("ui_hint", "")).strip()
            if not coach_text:
                return None
            if not ui_hint:
                ui_hint = "保持穩定"
            return {"coach_text": coach_text, "ui_hint": ui_hint}
        except Exception:
            return None

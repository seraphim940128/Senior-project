# Project/main - Rehab Coach Initial Architecture

依 `計劃書.md` 建立的初步整合版本，採三層流程：

1. Layer 1：YOLO + Transformer 動作辨識（含 `is_stable`/`segment_id`）
2. Layer 2：MediaPipe 關節點評估（輸出 `metrics` + `posture_summary`）
3. Layer 3：回饋生成（模板 + 可選 LLM），輸出 `coach_text` / `ui_hint`

## Directory

```text
Project/main/
  best_transformer_model_25_topk.pt
  yolo11n.pt
  run_rehab_coach.py
  rehab_coach/
    __init__.py
    config.py
    models.py
    layer1_action.py
    layer2_pose.py
    layer3_feedback.py
    pipeline.py
    unity_socket.py
```

## Included Weights

- 動作分類權重：`Project/main/best_transformer_model_25_topk.pt`
- YOLO 權重：`Project/main/yolo11n.pt`

## Supported Actions (7)

- `elbow_flexion_left`
- `elbow_flexion_right`
- `shoulder_flexion_left`
- `shoulder_flexion_right`
- `shoulder_abduction_left`
- `shoulder_abduction_right`
- `shoulder_forward_elevation`

注意：模型原始類別仍含 `side_tap_left/right`，但此架構已將 side tap 排除，不會進入 stable segment。

## Run

在 `Project/main` 執行：

```bash
python run_rehab_coach.py --src 0 --device auto
```

## Dependencies

```bash
pip install opencv-python mediapipe ultralytics torch torchvision numpy
```

可選參數：

```bash
python run_rehab_coach.py ^
  --src 0 ^
  --stable-frames 15 ^
  --stable-conf 0.8 ^
  --cooldown 1.8 ^
  --unity --unity-host 127.0.0.1 --unity-port 5500 ^
  --unity-conf 0.9 --unity-cooldown 3.0 ^
  --use-llm --llm-model gpt-4.1-mini
```

若你的 mediapipe 版本只有 `tasks`（沒有 `mp.solutions.pose`），可額外指定：

```bash
python run_rehab_coach.py --src 0 --pose-task-model path\\to\\pose_landmarker.task
```

自我檢查（依賴 + 模型檔）：

```bash
python run_rehab_coach.py --self-check
```

## Unity 單字元控制格式

目前已改為舊版 `socket_usage.py` 相同格式：

- `shoulder_abduction_left -> u`
- `shoulder_abduction_right -> o`
- `shoulder_flexion_left -> j`
- `shoulder_flexion_right -> l`
- `shoulder_forward_elevation -> i`

當預測置信度達到 `unity-conf`，且超過 `unity-cooldown`，就會透過 TCP 傳送單一字元。

## Output Schema

每個 segment 結束時會輸出 JSON（與企劃書一致）：

```json
{
  "action": "shoulder_forward_elevation",
  "segment": {
    "id": 21,
    "duration_s": 1.0,
    "confidence_mean": 0.88
  },
  "metrics": {
    "primary_peak": 162.4,
    "primary_mean": 138.2,
    "primary_std": 5.1,
    "reach_ratio": 0.78,
    "comp_peak": 0.23,
    "comp_mean": 0.12,
    "symmetry_peak_diff": 14.2,
    "wrist_above_head_ratio": 0.62
  },
  "posture_summary": {
    "primary_joint_range": "insufficient",
    "compensation": "mild",
    "symmetry": "imbalanced",
    "stability": "stable"
  }
}
```

## Reference

Layer 2 的動作邏輯參考：`Project/else/posture_detection.py`、`Project/else/pose_dection.py`。

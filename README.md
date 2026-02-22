# Rehab Coach (Project/main)

即時復健動作辨識與姿勢回饋管線，整合：

1. Layer 1：YOLO + MobileNet 特徵 + Transformer 動作分類
2. Layer 2：MediaPipe 姿勢估測與動作品質評估（角度、代償、穩定度）
3. Layer 3：規則式 / LLM 中文回饋文字
4. 可選 Unity TCP 單字元指令輸出

主要入口程式為 `run_rehab_coach.py`。

## 功能摘要

- 即時相機或影片輸入（`--src 0` / `--src path/to/video`）
- 動作穩定段（stable segment）切分與摘要輸出（JSON）
- 動作品質評估（ROM、代償、對稱、穩定度）
- 指定目標動作計數（`--target-action`）
- Unity TCP 控制（可設定門檻與 cooldown）
- 可選 OpenAI LLM 回饋（`--use-llm`）
- 視窗預覽與文字疊圖（可用 `--no-window` 關閉）
- 啟動前自我檢查（`--self-check`）

## 資料夾結構（重點）

```text
Project/main/
├─ run_rehab_coach.py                  # 主程式
├─ README.md
├─ yolo11n.pt                          # YOLO 權重
├─ best_transformer_model_25_topk.pt   # 主要動作分類權重
├─ best_transformer_model_max_fold_1.pt # 舊版/實驗腳本使用權重
├─ pose_landmarker_heavy.task          # MediaPipe Tasks 模型（可選）
├─ rehab_coach/
│  ├─ __init__.py
│  ├─ config.py                        # labels、支援動作、閾值、baseline
│  ├─ models.py                        # Transformer classifier 與權重載入
│  ├─ layer1_action.py                 # Layer1 動作辨識
│  ├─ layer2_pose.py                   # Layer2 姿勢抽取與評估
│  ├─ layer3_feedback.py               # Layer3 回饋（規則式/LLM）
│  ├─ pipeline.py                      # 三層流程串接
│  └─ unity_socket.py                  # Unity TCP client
├─ test_video/                         # 測試影片範例
├─ socket_usage.py                     # 舊版/原型腳本（Unity）
├─ socket_usage_v2.py                  # 舊版/原型腳本（Unity）
├─ socket_usage_v3.py                  # 舊版/原型腳本（Unity）
└─ socket_test.py                      # 手動發送 TCP 指令測試
```

## 支援動作

目前主流程（stable segment / Layer2 評估）支援 7 種：

- `elbow_flexion_left`
- `elbow_flexion_right`
- `shoulder_flexion_left`
- `shoulder_flexion_right`
- `shoulder_abduction_left`
- `shoulder_abduction_right`
- `shoulder_forward_elevation`

模型分類標籤共 9 種，另外包含：

- `side_tap_left`
- `side_tap_right`

注意：`side_tap_*` 目前在 `run_rehab_coach.py` 主流程中不列為 `supported_actions`，因此不會形成 Layer2 摘要段落。

## 安裝需求

建議 Python `3.9+`。

### 必要套件

```bash
pip install opencv-python mediapipe ultralytics torch torchvision numpy pillow
```

### 可選套件

- 啟用 LLM 回饋（`--use-llm`）：

```bash
pip install openai
```

- 使用 `socket_test.py` 鍵盤測試：

```bash
pip install keyboard
```

## 快速開始

在 `Project/main` 目錄執行：

```bash
python run_rehab_coach.py --src 0 --device auto
```

使用影片測試：

```bash
python run_rehab_coach.py --src "test_video/shoulder_forward_elevation_80.MOV" --device cpu
```

指定目標動作並計數：

```bash
python run_rehab_coach.py --src 0 --target-action shoulder_forward_elevation
```

啟用 Unity 與 LLM（範例）：

```bash
python run_rehab_coach.py ^
  --src 0 ^
  --unity --unity-host 127.0.0.1 --unity-port 5500 ^
  --unity-conf 0.9 --unity-cooldown 3.0 ^
  --use-llm --llm-model gpt-5-mini
```

執行自我檢查（依賴與權重檔存在檢查）：

```bash
python run_rehab_coach.py --self-check
```

無視窗模式（例如遠端/錄製環境）：

```bash
python run_rehab_coach.py --src 0 --no-window
```

## CLI 參數（`run_rehab_coach.py`）

- `--src`：相機 index 或影片路徑（預設 `0`）
- `--target-action`：指定目標動作名稱（成功次數統計）
- `--weights`：動作分類權重 `.pt` 路徑
- `--yolo`：YOLO 權重 `.pt` 路徑
- `--device`：`auto|cpu|cuda`
- `--stable-frames`：穩定段判定所需連續幀數（覆蓋 config）
- `--stable-conf`：動作置信度門檻（覆蓋 config）
- `--cooldown`：Layer3 回饋事件 cooldown 秒數（覆蓋 config）
- `--unity`：啟用 Unity TCP 輸出
- `--unity-host` / `--unity-port`：Unity TCP 位址
- `--unity-conf`：發送 Unity 指令置信度門檻
- `--unity-cooldown`：Unity 指令 cooldown 秒數
- `--use-llm`：啟用 Layer3 LLM 回饋（需 `OPENAI_API_KEY`）
- `--llm-model`：LLM 模型名稱（預設 `gpt-5-mini`）
- `--pose-task-model`：MediaPipe Tasks `.task` 模型路徑（可選）
- `--self-check`：檢查依賴與模型檔後結束
- `--no-window`：關閉 OpenCV 視窗

## 輸出內容（執行時）

主程式在偵測到穩定動作段落結束後，會輸出一筆 JSON 摘要（summary）：

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

若觸發回饋事件（姿勢品質不佳且超過 cooldown），也會輸出：

```json
{
  "action": "shoulder_forward_elevation",
  "coach_text": "…",
  "ui_hint": "…"
}
```

## Unity TCP 對接

`run_rehab_coach.py` 使用 `rehab_coach/unity_socket.py`，每次連線送出 1 個字元命令（TCP）。

目前主流程對應如下：

- `shoulder_abduction_left` -> `u`
- `shoulder_abduction_right` -> `o`
- `shoulder_flexion_left` -> `j`
- `shoulder_flexion_right` -> `l`
- `shoulder_forward_elevation` -> `i`

發送條件：

- 啟用 `--unity`
- 動作在 `unity_action_map` 中
- 預測信心值 >= `unity_confidence_threshold`（預設 `0.90`）
- 距離上次發送時間 >= `unity_command_cooldown_s`（預設 `3.0s`）

## LLM 回饋（可選）

若使用 `--use-llm`，請先設定環境變數（PowerShell）：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

若未安裝 `openai` 套件或未設定 API key，程式會回退到規則式模板回饋。

## 已知注意事項

- `PoseExtractor` 目前會優先使用 `mediapipe.solutions.pose`；只有在該後端不可用時才會嘗試 `--pose-task-model`（Tasks backend）。
- `README` 已依現有程式碼更新；`socket_usage*.py` 與 `socket_test.py` 為原型/測試腳本，與主流程設定可能不同。
- 專案含大型模型與測試影片檔案，版本控管時建議注意檔案大小與 `.gitignore`（如 `__pycache__/`）。


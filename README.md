# Rehab Coach (`Project/main`)

以攝影機或影片輸入進行復健動作辨識、姿勢評估與回饋的原型專案。

目前主要入口為 `run_rehab_coach.py`，流程分為：

1. `Layer 1`：YOLO 人體偵測 + MobileNet 特徵提取 + Transformer 動作分類
2. `Layer 2`：MediaPipe 姿勢點擷取與運動學指標評估（ROM / 代償 / 穩定度）
3. `Layer 3`：回饋文字生成（模板為主，可選 LLM）
4. `Unity TCP`：將特定動作映射為單字元指令傳送給 Unity

## 專案結構

```text
Project/main/
├─ .git/                              # Git 版本控制資料夾
├─ .gitignore                         # Python/IDE/暫存忽略規則
├─ README.md                          # 本說明文件
├─ run_rehab_coach.py                 # 主程式（即時流程 / CLI）
├─ temp                               # 空白暫存檔（0 bytes）
├─ yolo11n.pt                         # YOLO 權重
├─ best_transformer_model_25_topk.pt  # 動作分類器權重（主流程預設）
├─ best_transformer_model_max_fold_1.pt # 舊版/實驗腳本使用權重
├─ pose_landmarker_heavy.task         # MediaPipe Tasks PoseLandmarker 模型（可選）
├─ socket_test.py                     # 手動送 Unity TCP 指令測試（鍵盤）
├─ socket_usage.py                    # 舊版單檔動作辨識 + Unity 示範
├─ socket_usage_v2.py                 # 舊版示範（不同權重/ROI 設定）
├─ socket_usage_v3.py                 # 舊版示範（不同 Unity 動作映射）
├─ rehab_coach/
│  ├─ __init__.py
│  ├─ config.py                       # 模型路徑、閾值、動作清單、基準參數
│  ├─ models.py                       # Transformer 分類器與特徵緩衝
│  ├─ layer1_action.py                # Layer 1 動作辨識
│  ├─ layer2_pose.py                  # Layer 2 姿勢點與運動學評估
│  ├─ layer3_feedback.py              # Layer 3 回饋文字生成（模板/LLM）
│  ├─ pipeline.py                     # Layer1 + Layer2 管線整合
│  └─ unity_socket.py                 # Unity TCP client
└─ test_video/
   ├─ elbow_flexion_right_75.MOV
   ├─ elbow_flexion_right_76.MOV
   ├─ elbow_flexion_right_77.MOV
   ├─ shoulder_abduction_left_79.MOV
   ├─ shoulder_abduction_left_80.MOV
   ├─ shoulder_abduction_right_80.MOV
   ├─ shoulder_flexion_left_78.MOV
   ├─ shoulder_flexion_left_79.MOV
   ├─ shoulder_flexion_left_80.MOV
   ├─ shoulder_flexion_right_80.MOV
   └─ shoulder_forward_elevation_80.MOV
```

## 支援動作（主流程）

`rehab_coach/config.py` 目前 `supported_actions` 共 7 類：

- `elbow_flexion_left`
- `elbow_flexion_right`
- `shoulder_flexion_left`
- `shoulder_flexion_right`
- `shoulder_abduction_left`
- `shoulder_abduction_right`
- `shoulder_forward_elevation`

分類模型標籤 (`model_labels`) 共 9 類，另外包含：

- `side_tap_left`
- `side_tap_right`

這兩類可被模型辨識，但主流程不會進入 Layer 2 姿勢評估與計數邏輯。

## 環境需求

- Python `3.9+`（建議）
- Windows（目前 README 例子以 PowerShell / Windows 路徑為主）

## 安裝

### 核心依賴

```bash
pip install opencv-python mediapipe ultralytics torch torchvision numpy pillow
```

### 可選依賴

- 使用 LLM 回饋（`--use-llm`）：

```bash
pip install openai
```

- 執行 `socket_test.py` 鍵盤測試：

```bash
pip install keyboard
```

## 快速開始

在 `Project/main` 目錄下執行：

### 1. 攝影機即時辨識

```bash
python run_rehab_coach.py --src 0 --device auto
```

### 2. 測試影片

```bash
python run_rehab_coach.py --src "test_video/shoulder_forward_elevation_80.MOV" --device cpu
```

### 3. 指定目標動作（計數模式）

```bash
python run_rehab_coach.py --src 0 --target-action shoulder_forward_elevation
```

### 4. 啟用 Unity TCP 與 LLM（可選）

```powershell
python run_rehab_coach.py `
  --src 0 `
  --unity --unity-host 127.0.0.1 --unity-port 5500 `
  --unity-conf 0.9 --unity-cooldown 3.0 `
  --use-llm --llm-model gpt-5-mini `
  --pose-task-model pose_landmarker_heavy.task
```

### 5. 自我檢查（依賴與模型檔）

```bash
python run_rehab_coach.py --self-check
```

### 6. 不開 OpenCV 視窗（純輸出模式）

```bash
python run_rehab_coach.py --src 0 --no-window
```

## CLI 參數（`run_rehab_coach.py`）

- `--src`：攝影機 index（如 `0`）或影片路徑
- `--target-action`：指定要追蹤/計數的動作名稱
- `--weights`：動作分類模型 `.pt` 路徑
- `--yolo`：YOLO 權重 `.pt` 路徑
- `--device`：`auto | cpu | cuda`
- `--stable-frames`：覆寫穩定判定幀數（預設取 `config`）
- `--stable-conf`：覆寫動作分類信心閾值（預設取 `config`）
- `--cooldown`：覆寫 Layer3 回饋冷卻秒數（預設取 `config`）
- `--unity`：啟用 Unity TCP 指令輸出
- `--unity-host` / `--unity-port`：Unity TCP 位址與埠號
- `--unity-conf`：Unity 指令送出最低信心值
- `--unity-cooldown`：Unity 指令冷卻秒數
- `--use-llm`：啟用 Layer3 LLM 回饋（需設定 `OPENAI_API_KEY`）
- `--llm-model`：LLM 模型名稱（預設 `gpt-5-mini`）
- `--pose-task-model`：MediaPipe Tasks `.task` 模型路徑（可選）
- `--self-check`：檢查依賴與必要權重檔後結束
- `--no-window`：不顯示 OpenCV 視窗

## 輸出內容（實際欄位）

當一段穩定動作結束時，主程式會輸出 Layer2 `summary` JSON（若片段長度 `< 1.0s` 會略過）。

### `summary` 範例

```json
{
  "action": "shoulder_forward_elevation",
  "segment": {
    "id": 21,
    "duration_s": 1.234,
    "confidence_mean": 0.8812
  },
  "metrics": {
    "primary_peak": 162.4,
    "primary_mean": 138.2,
    "primary_std": 5.1,
    "reach_ratio": 0.78,
    "comp_peak": 6.3,
    "comp_mean": 3.2,
    "sparc": -5.8,
    "rms_acc": 7.1
  },
  "posture_summary": {
    "primary_joint_range": "acceptable",
    "compensation": "mild",
    "symmetry": "not_applicable",
    "stability": "stable"
  }
}
```

### Layer3 回饋事件範例

當 `posture_summary` 判定為需要回饋（活動度不足 / 代償過大 / 不穩定等）且超過冷卻時間時，會輸出：

```json
{
  "action": "shoulder_forward_elevation",
  "coach_text": "請放慢速度並保持軀幹穩定。",
  "ui_hint": "放慢節奏，避免抖動"
}
```

## 計分與計數規則（主程式目前邏輯）

`run_rehab_coach.py` 會根據 `posture_summary` 計分：

- 活動度（ROM）：`good=50`、`acceptable=35`、`insufficient=10`
- 代償（compensation）：`none=30`、`mild=15`、`excessive=0`
- 穩定度（stability）：`stable=20`、`unstable=0`

總分 `>= 60` 視為成功一次動作。

- 有設定 `--target-action`：只統計目標動作成功次數
- 未設定 `--target-action`：統計所有支援動作的成功次數

## Unity TCP 指令映射（主流程）

`rehab_coach/config.py` 目前映射如下：

- `shoulder_abduction_left` -> `u`
- `shoulder_abduction_right` -> `o`
- `shoulder_flexion_left` -> `j`
- `shoulder_flexion_right` -> `l`
- `shoulder_forward_elevation` -> `i`

送出條件（主流程）：

- 啟用 `--unity`
- 預測動作存在於映射表
- 預測信心值 `>= unity_confidence_threshold`（預設 `0.90`）
- 距離上次送出時間 `>= unity_command_cooldown_s`（預設 `3.0s`）

## MediaPipe 後端說明

`PoseExtractor` 會依環境選擇姿勢擷取後端：

1. 優先使用 `mediapipe.solutions.pose`（若可用）
2. 若提供 `--pose-task-model` 且檔案存在，則嘗試使用 MediaPipe Tasks `PoseLandmarker`
3. 若皆不可用，Layer2 將無法產出姿勢評估摘要

## 舊版測試腳本說明

- `socket_usage.py`：早期單檔版動作辨識 + Unity 指令送出示範
- `socket_usage_v2.py`：舊版變體（不同權重與特徵尺寸）
- `socket_usage_v3.py`：舊版變體（示範 `side_tap` / `elbow_flexion` 指令映射）
- `socket_test.py`：純 TCP 手動按鍵測試（`i/j/l/u/o`）

這些腳本可作為驗證與對照用途；建議主要使用 `run_rehab_coach.py`。

## `.gitignore` 重點

已包含常見忽略規則：

- Python 快取：`__pycache__/`, `*.pyc`
- 虛擬環境：`.venv/`, `venv/`
- 測試與工具快取：`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- 日誌與暫存：`*.log`, `*.tmp`, `*.temp`
- IDE 設定：`.vscode/`, `.idea/`
- `ultralytics` 常見輸出：`runs/`

另外已保留可選註解（取消註解即可忽略大型模型/影片檔）：

- `*.pt`, `*.task`, `*.MOV`, `*.mp4`, `*.avi`


# Rehab Coach

---

## 系統架構概覽

主程式入口為 **`run_rehab_coach.py`**，整體流程分為四層：

| 層級 | 名稱 | 說明 |
| --- | --- | --- |
| **Layer 1** | 動作辨識 | 使用 YOLO 偵測人體 → 以 MobileNet 擷取特徵 → 以 Transformer 進行動作分類，輸出當前最可能的動作標籤與信心度。 |
| **Layer 2** | 姿勢評估 | 使用 MediaPipe 擷取關節點（landmarks），計算運動學指標：主要關節活動度（ROM）、代償行為（軀幹傾斜、聳肩等）、動作穩定度（SPARC、RMS 加速度）。 |
| **Layer 3** | 回饋生成 | 依 Layer2 的評估結果產生教練式文字回饋；可選用模板或 LLM（如 GPT）生成。 |
| **Unity TCP** | 指令輸出 | 將辨識到的特定動作映射為單字元指令，透過 TCP 傳送給 Unity 端，用於驅動 3D 示範或遊戲化介面。 |

---

## 專案目錄結構

```text
Project/main/
├─ .git/                              # Git 版本控制
├─ .gitignore                         # 忽略規則（Python / IDE / 暫存檔）
├─ README.md                          # 本說明文件
├─ run_rehab_coach.py                 # 主程式：即時辨識流程與 CLI 介面
├─ temp                               # 空白暫存檔（0 bytes）
├─ yolo11n.pt                         # YOLO 人體偵測權重
├─ best_transformer_model_25_topk.pt  # 動作分類器權重（主流程預設）
├─ best_transformer_model_max_fold_1.pt # 舊版／實驗用權重
├─ pose_landmarker_heavy.task         # MediaPipe Tasks 姿勢模型（可選，較高精度）
├─ socket_test.py                     # 手動送 Unity TCP 指令測試（鍵盤觸發）
├─ socket_usage.py                    # 舊版單檔動作辨識 + Unity 示範
├─ socket_usage_v2.py                 # 舊版示範（不同權重／ROI 設定）
├─ socket_usage_v3.py                 # 舊版示範（不同 Unity 動作映射）
├─ rehab_coach/                        # 核心套件
│  ├─ __init__.py
│  ├─ config.py                       # 模型路徑、閾值、支援動作清單、基準參數
│  ├─ models.py                       # Transformer 分類器與特徵緩衝區
│  ├─ layer1_action.py                # Layer 1：動作辨識
│  ├─ layer2_pose.py                  # Layer 2：姿勢點擷取與運動學評估
│  ├─ layer3_feedback.py              # Layer 3：回饋文字生成（模板／LLM）
│  ├─ pipeline.py                     # Layer1 + Layer2 管線整合與片段管理
│  └─ unity_socket.py                 # Unity TCP 客戶端

```

---

## 支援的動作類型

在 **`rehab_coach/config.py`** 中定義：

- **`supported_actions`**（共 7 類）：主流程會對這些動作進行 Layer2 姿勢評估與計分／計數。
- **`model_labels`**（共 9 類）：分類模型實際輸出的標籤集合，比支援動作多出 `side_tap_left`、`side_tap_right`；此二類可被辨識，但**不會**進入 Layer2 評估與主程式的成功計數邏輯。

### 支援動作一覽

| 動作名稱 | 說明（對應中文／用途） |
| --- | --- |
| `elbow_flexion_left` | 左肘屈曲 |
| `elbow_flexion_right` | 右肘屈曲 |
| `shoulder_flexion_left` | 左肩屈曲 |
| `shoulder_flexion_right` | 右肩屈曲 |
| `shoulder_abduction_left` | 左肩外展 |
| `shoulder_abduction_right` | 右肩外展 |
| `shoulder_forward_elevation` | 雙手上舉至頭頂 |

---

## 環境需求

- **Python**：建議 3.9 以上
- **作業系統**：本說明以 Windows（PowerShell）為例，路徑與指令皆可依實際環境調整
- **硬體**：使用 GPU 時需支援 CUDA；僅 CPU 亦可執行，速度較慢

---

## 安裝步驟

### 1. 核心依賴（必要）

```bash
pip install opencv-python mediapipe ultralytics torch torchvision numpy pillow
```

上述套件用於影像讀取、人體偵測、姿勢估計、動作分類與特徵處理。

### 2. 可選依賴

- **啟用 Layer3 LLM 回饋**（`--use-llm`）時需安裝：

  ```bash
  pip install openai
  ```

  並在環境變數中設定 `OPENAI_API_KEY`。

- **執行 `socket_test.py` 鍵盤測試**時需安裝：

  ```bash
  pip install keyboard
  ```

---

## 快速開始

請在 **`Project/main`** 目錄下執行以下指令。

### 1. 使用攝影機即時辨識

```bash
python run_rehab_coach.py --src 0 --device auto
```

- `--src 0`：使用預設攝影機（通常為 0）
- `--device auto`：自動選擇 CPU 或 CUDA

### 2. 使用測試影片

```bash
python run_rehab_coach.py --src "test_video/shoulder_forward_elevation_80.MOV" --device cpu
```

適合無攝影機時驗證流程或除錯；使用影片時可指定 `--device cpu` 以節省 GPU 資源。

### 3. 指定目標動作（僅計數單一動作）

```bash
python run_rehab_coach.py --src 0 --target-action shoulder_forward_elevation
```

系統只會將「辨識為 `shoulder_forward_elevation` 且達成功標準」的次數計入，並在畫面上顯示該動作的完成次數。

### 4. 啟用 Unity TCP 與 LLM 回饋（進階）

```powershell
python run_rehab_coach.py `
  --src 0 `
  --unity --unity-host 127.0.0.1 --unity-port 5500 `
  --unity-conf 0.9 --unity-cooldown 3.0 `
  --use-llm --llm-model gpt-5-mini `
  --pose-task-model pose_landmarker_heavy.task
```

- `--unity`：開啟 TCP 連線，將動作指令送往 Unity
- `--unity-conf`、`--unity-cooldown`：控制指令送出之信心閾值與冷卻時間
- `--use-llm`、`--llm-model`：啟用 LLM 生成文字回饋
- `--pose-task-model`：使用 MediaPipe Tasks 較重模型，可提升關節點精度

### 5. 自我檢查（依賴與模型檔是否存在）

```bash
python run_rehab_coach.py --self-check
```

會檢查必要 Python 套件以及 YOLO、動作分類權重等檔案是否存在，適合首次安裝或部署時使用。

### 6. 不顯示 OpenCV 視窗（純後端／遠端執行）

```bash
python run_rehab_coach.py --src 0 --no-window
```

影像仍會經由管線處理，僅不顯示即時預覽視窗。

---

## 命令列參數總覽（`run_rehab_coach.py`）

| 參數 | 說明 | 預設或備註 |
| --- | --- | --- |
| `--src` | 影像來源：攝影機 index（如 `0`）或影片檔案路徑 | 必填（或依腳本預設） |
| `--target-action` | 指定要追蹤與計數的動作名稱 | 不指定則統計所有支援動作 |
| `--weights` | 動作分類模型 `.pt` 路徑 | 預設為 `best_transformer_model_25_topk.pt` |
| `--yolo` | YOLO 權重 `.pt` 路徑 | 預設為 `yolo11n.pt` |
| `--device` | 運算裝置 | `auto`（自動）、`cpu`、`cuda` |
| `--stable-frames` | 連續多少幀判定為「穩定」才開始／維持片段 | 預設由 config 讀取（如 3） |
| `--stable-conf` | 動作分類信心閾值，高於此才計入穩定計數 | 預設由 config 讀取（如 0.65） |
| `--cooldown` | Layer3 回饋冷卻時間（秒），避免短時間內重複提示 | 預設由 config 讀取（如 1.8） |
| `--unity` | 是否啟用 Unity TCP 輸出 | 預設不啟用 |
| `--unity-host` | Unity TCP 主機位址 | 預設 `127.0.0.1` |
| `--unity-port` | Unity TCP 埠號 | 預設 `5500` |
| `--unity-conf` | 送出 Unity 指令所需之最低信心值 | 預設 0.9 |
| `--unity-cooldown` | Unity 指令冷卻時間（秒） | 預設 3.0 |
| `--use-llm` | 是否啟用 Layer3 LLM 回饋 | 需設定 `OPENAI_API_KEY` |
| `--llm-model` | LLM 模型名稱 | 預設 `gpt-5-mini` |
| `--pose-task-model` | MediaPipe Tasks 姿勢模型 `.task` 路徑 | 不指定則使用 legacy MediaPipe |
| `--self-check` | 僅執行依賴與檔案檢查後結束 | 不跑主流程 |
| `--no-window` | 不顯示 OpenCV 預覽視窗 | 適合無顯示環境 |

---

## 輸出內容說明

當一段**穩定動作片段結束**時，主程式會輸出 Layer2 的 **`summary`** JSON。若片段**持續時間少於 1.0 秒**，會視為雜訊而**不計入**統計與回饋。

### `summary` 結構範例

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
    "trunk_dev_max": 6.3,
    "sparc": -5.8,
    "rms_acc": 7.1
  },
  "posture_summary": {
    "primary_joint_range": "acceptable",
    "compensation": "none",
    "symmetry": "not_applicable",
    "stability": "stable"
  }
}
```

- **`action`**：辨識到的動作名稱。
- **`segment`**：片段編號、持續時間（秒）、該片段內分類信心度平均。
- **`metrics`**：運動學數值（主要角度峰值／平均／標準差、達成 ROM 比例、軀幹偏差、SPARC 平滑度、RMS 加速度等）。
- **`posture_summary`**：彙總判定（關節活動度是否足夠、有無代償、對稱性、穩定度），供計分與 Layer3 回饋使用。

### Layer3 回饋事件範例

當系統判定需要給予教練回饋（例如活動度不足、代償過大、不穩定）且已超過回饋冷卻時間時，會輸出回饋事件，例如：

```json
{
  "action": "shoulder_forward_elevation",
  "coach_text": "請放慢速度並保持軀幹穩定。",
  "ui_hint": "放慢節奏，避免抖動"
}
```

`coach_text`、`ui_hint` 可依實作用於畫面或語音提示。

---

## 計分與計數規則（主程式邏輯）

`run_rehab_coach.py` 會依 **`posture_summary`** 計算單次動作總分，並決定是否計入「成功次數」：

| 項目 | 判定 | 得分 |
| --- | --- | --- |
| 關節活動度（ROM） | `good` | 50 |
| | `acceptable` | 35 |
| | `insufficient` | 10 |
| 代償行為 | `none` | 30 |
| | 有 1 項代償 | 15 |
| | 多項代償 | 0 |
| 穩定度 | `stable` | 20 |
| | `unstable` | 0 |

- **總分 ≥ 60** 視為該次動作**成功**，會計入次數。
- **有設定 `--target-action`**：只統計「辨識為該目標動作且成功」的次數。
- **未設定 `--target-action`**：統計所有支援動作各自的成功次數。

---

## Unity TCP 指令映射

主流程中，下列動作會對應到單字元指令並在符合條件時透過 TCP 送出（設定見 **`rehab_coach/config.py`**）：

| 動作名稱 | 送出字元 | 說明（示意） |
| --- | --- | --- |
| `shoulder_abduction_left` | `u` | 左肩外展 |
| `shoulder_abduction_right` | `o` | 右肩外展 |
| `shoulder_flexion_left` | `j` | 左肩屈曲 |
| `shoulder_flexion_right` | `l` | 右肩屈曲 |
| `shoulder_forward_elevation` | `i` | 雙手前舉過頭 |

**送出條件**（須同時滿足）：

- 已啟用 `--unity`
- 當前預測動作存在於上述映射表
- 預測信心度 ≥ `unity_confidence_threshold`（預設 0.90）
- 距離上次送出已超過 `unity_command_cooldown_s`（預設 3.0 秒）

Unity 端需自行實作 TCP 伺服器，接收這些字元並對應到動畫或介面邏輯。

---

## MediaPipe 姿勢擷取後端

**`PoseExtractor`**（在 `rehab_coach/layer2_pose.py`）會依環境選擇姿勢估計方式：

1. **優先**：若已安裝 `mediapipe` 且具 `solutions.pose`，則使用 **MediaPipe Solutions**（legacy）擷取關節點。
2. **可選**：若命令列提供 **`--pose-task-model`** 且對應 `.task` 檔案存在，則嘗試改用 **MediaPipe Tasks** 的 `PoseLandmarker`（例如 `pose_landmarker_heavy.task`），通常可得到較穩定的關節點，有利於 Layer2 角度與穩定度計算。
3. 若上述皆無法使用，Layer2 將無法產出姿勢評估摘要，主流程仍可跑 Layer1 動作辨識，但不會有 ROM／代償／穩定度等輸出。

---

## 舊版與測試腳本

| 腳本 | 用途 |
| --- | --- |
| `socket_usage.py` | 早期單檔版：動作辨識 + 送 Unity 指令示範 |
| `socket_usage_v2.py` | 舊版變體（不同權重或 ROI 設定） |
| `socket_usage_v3.py` | 舊版變體（不同動作與 Unity 字元映射） |
| `socket_test.py` | 不跑辨識，僅以鍵盤手動送 TCP 指令（如 `i`、`j`、`l`、`u`、`o`） |

---

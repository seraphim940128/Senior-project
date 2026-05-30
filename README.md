# Posture Correction

> 動作感知的即時復健姿態評估管線 —— 以單一狀態機切分每一下動作，由多個獨立評估器
> 評分（ROM / phase_rom / symmetry / compensation），並產生 0–100 分與可解釋的回饋。

動作**預先選定**（無動作分類器）。攝影機 / 影片影格經由共用的 `LiveSession` 管線，
被唯一的線上狀態機 `PhaseTracker` 切分成「次數（repetition）」；每完成一次數即由各
評估器評分、加權彙總並產生回饋。**同一份每幀快照同時驅動即時疊圖與正式評分** ——
螢幕上畫的曲線，就是真正被評分的數據。

---

## 核心功能特色（Features）

- **四面向評分**
  - `rom`：關節活動度（ROM）是否達到該動作的最低標準。
  - `phase_rom`：上升段與下降段是否各自完整（抓「快速亂揮」「下放失控」）。
  - `symmetry`：雙臂前舉的幅度（LSI）與起動時序是否對稱（僅 `shoulder_forward_elevation`）。
  - `compensation`：軀幹側傾（TLS）+ 肩部上抬（SE）的規則式代償偵測。
  - 另含 `dtw` / `deviation` 兩個**已接線、未實作**的架構槽位。
- **唯一狀態機**：全專案只有一個 `PhaseTracker`（測試強制），即時與離線結果一致。
- **個人化代償基線**：量「相對個人休息姿勢的偏移」，吸收體型與鏡頭傾斜差異；資料不足
  時誠實回報 `unreliable` 而非硬猜。
- **座標基準分流**：ROM / 切分用 2D（正面攝影機）、symmetry / compensation 用 3D
  （避免透視縮短造成假性差異）。
- **門檻全外置**：所有數值門檻只在 `config/default.yaml`（測試強制原始碼不得硬編碼）；
  ROM 參考值附文獻出處。
- **兩種使用形態**：無頭跑分輸出 JSON；四個即時 demo 含影像 + 圖表雙視窗。

---

## 技術堆疊與依賴（Tech Stack）

| 類別 | 套件 | 版本 |
|------|------|------|
| 語言 | Python | `>= 3.10` |
| 核心 | NumPy / PyYAML | `>= 1.26` / `>= 6.0` |
| 視覺（選用 `vision`） | MediaPipe / OpenCV | `>= 0.10` / `>= 4.10` |
| 測試（選用 `dev`） | pytest | `>= 8.0` |

> 核心管線、評估器與測試**不需** MediaPipe / OpenCV 即可執行（`cv2` 為延遲匯入）；
> 只有實際讀取攝影機 / 影片與顯示視窗時才需要 `vision` extra。

---

## 安裝與啟動步驟（Installation & Getting Started）

```bash
# 於 posture-correction/ 目錄下，建議使用虛擬環境
pip install -e .[dev,vision]
```

安裝後會註冊 console script：`posture-correction`（對應 `app:main`）。

---

## 使用方式與範例（Usage / Examples）

### 1) 正式跑分（無頭，輸出 JSON）

對一段影片或攝影機、針對一個預選動作評分，將每次數的評估結果寫到
`data/session_outputs/`：

```bash
# 影片
posture-correction --action shoulder_flexion_right --src clip.mp4

# 攝影機（index 0）；按 R 校正休息姿勢、q / ESC 結束
posture-correction --action shoulder_flexion_right --src 0

# 指定輸出檔
posture-correction --action shoulder_flexion_right --src clip.mp4 -o out.json

# 附上每幀代償診斷剖面（除錯用）
posture-correction --action shoulder_forward_elevation --src clip.mp4 --debug

posture-correction --action shoulder_flexion_right --src 0 -o out.json --debug
```

`--action`（必填）可選值：
`elbow_flexion_left/right`、`shoulder_flexion_left/right`、
`shoulder_abduction_left/right`、`shoulder_forward_elevation`。

> 影片模式為自動校正（從開頭有效幀取休息基線）；攝影機模式為手動校正（按 R）。

### 2) 即時 demo（影像 + 圖表雙視窗）

```bash
python -m live.apps.live_rom shoulder_flexion_right        # ROM 進度條
python -m live.apps.live_phase_rom shoulder_flexion_right  # 上升/峰值/下降相位
python -m live.apps.live_symmetry                          # 固定 shoulder_forward_elevation
python -m live.apps.live_compensation shoulder_abduction_right  # 代償，按 R 校正
```

即時視窗快捷鍵：`q` / `ESC` 結束、`r` 重置（或重新校正）、`s` 截圖至 `screenshots/`。

### 3) 執行測試

```bash
python -m pytest tests/ -q
```

### JSON 輸出格式（節錄）

```jsonc
{
  "project": "posture-correction",
  "action": "shoulder_flexion_right",
  "source": "clip.mp4",
  "generated_at": "2026-05-26T00:00:00Z",
  "is_complete": true,
  "processed_frames": 1200,
  "pose_basis": "2d",
  "compensation_basis": "3d",
  "rep_count": 5,
  "reps": [
    {
      "rep_index": 1,
      "overall_score": 87.5,
      "metrics": [
        { "name": "rom", "status": "ok", "passed": true, "score": 100.0,
          "primary_value": 138.2, "detail": { "baseline_deg": 18.4, "peak_deg": 156.6,
          "required_min_deg": 132.5, "target_deg": 170.0 } },
        { "name": "phase_rom", "status": "ok", "passed": true, "score": 100.0, "detail": {} },
        { "name": "symmetry", "status": "not_applicable", "passed": null, "score": null },
        { "name": "compensation", "status": "ok", "passed": true, "score": 100.0, "detail": {} },
        { "name": "deviation", "status": "not_implemented", "passed": null, "score": null },
        { "name": "dtw", "status": "not_implemented", "passed": null, "score": null }
      ],
      "feedback": { "reason_codes": [], "correction_targets": [] }
    }
  ],
  "summary": { "rep_count": 5, "mean_overall_score": 85.1, "metric_pass_counts": { } }
}
```

`overall_score` 為加權平均（`rom`=2.0、`phase_rom`=1.0、`symmetry`=0.5、
`compensation`=1.0）；不適用 / 無參考 / 未實作的指標會退出分母重新正規化，而
`unreliable`（例如未校正）以 0 分計入。

---

## 專案目錄結構說明（Directory Structure）

> 僅列核心主程式、設定與功能模組；已略過測試（`tests/`、`test_*.py`）、log
> （`logs/`、`*.txt`）、編譯快取（`__pycache__/`）與打包產物（`*.egg-info/`）。

```text
posture-correction/
├── pyproject.toml                  # 套件中繼資料、相依、console script
├── README.md
├── config/
│   └── default.yaml                # 所有數值門檻（原始碼不得硬編碼）
├── data/
│   └── reference/
│       └── action_references.yaml  # 各動作 ROM 參考值（附文獻出處）
├── docs/                           # 技術文件
└── src/
    ├── app.py                      # 正式跑分 CLI 進入點（無頭 → JSON）
    ├── config.py                   # 讀取 YAML → 型別化 AppSettings
    ├── models.py                   # 純資料結構（dataclass / enum）
    ├── session_output.py           # 組裝 session JSON
    ├── pipeline/                   # 每幀流水線（物理量計算，可單元測試）
    │   ├── pose_detector.py        # MediaPipe 綁定（solutions / tasks 雙後端）
    │   ├── quality_filter.py       # landmark 可見度 / 存在度過濾
    │   ├── angle_extractor.py      # 動作主角度（三點角；前舉取 max(L,R)）
    │   ├── smoother.py             # 雙軌平滑（中位數物理軌 / EMA 顯示軌）
    │   ├── phase_tracker.py        # 唯一狀態機，發射 CompletedRepetition
    │   ├── posture_metrics.py      # 軀幹傾斜 / 肩髖位移幾何 proxy
    │   ├── compensation_baseline.py# 休息姿勢校正收集器與品質閘
    │   └── evaluators/
    │       ├── base.py             # MetricEvaluator 協定 + 未實作槽位
    │       ├── rom.py              # ROM
    │       ├── phase_rom.py        # 上升 / 下降相位 ROM
    │       ├── symmetry.py         # 對稱性（LSI + onset）
    │       ├── compensation.py     # 代償（TLS + SE）
    │       ├── dtw.py              # 槽位（未實作）
    │       └── deviation.py        # 槽位（未實作）
    ├── scoring/
    │   ├── aggregator.py           # 加權彙總 0–100
    │   └── feedback.py             # 回饋碼 + 矯正目標
    └── live/
        ├── session.py              # LiveSession：每幀管線編排（純邏輯）
        ├── app_runner.py           # 攝影機迴圈 + LiveMetricApp 協定
        ├── overlay_renderer.py     # 純 OpenCV 疊圖（零量測邏輯）
        └── apps/                   # 四個即時 demo
            ├── live_rom.py
            ├── live_phase_rom.py
            ├── live_symmetry.py
            └── live_compensation.py
```

---

## 已知限制

- 單一正面攝影機無法解析矢狀面（前後）傾斜。
- 代償為幾何 **proxy**（非真 3D 肩胛運動學）；門檻為文獻種子值，**尚未臨床驗證** ——
  屬研究級篩檢工具，非臨床量測儀器。
- 前舉 2D 會低估角度，故門檻已對 2D 量測值重新調校；symmetry / compensation 維持 3D。

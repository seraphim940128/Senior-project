"""
Source: yolo11_detect_person_v4.py
加上 label smoothing 並對訓練資料加上 frame flipping
"""

import os
import cv2
import csv
import json
import hashlib
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === 1. 動作類別 ===
LABELS = [
    "shoulder_abduction_left", "shoulder_abduction_right",
    "shoulder_flexion_left", "shoulder_flexion_right",
    "side_tap_left", "side_tap_right",
    "elbow_flexion_left", "elbow_flexion_right",
    "shoulder_forward_elevation"
]
label2idx = {label: i for i, label in enumerate(LABELS)}

FLIP_MAP = {
 "shoulder_abduction_left":"shoulder_abduction_right",
 "shoulder_abduction_right":"shoulder_abduction_left",
 "shoulder_flexion_left":"shoulder_flexion_right",
 "shoulder_flexion_right":"shoulder_flexion_left",
 "side_tap_left":"side_tap_right",
 "side_tap_right":"side_tap_left",
 "elbow_flexion_left":"elbow_flexion_right",
 "elbow_flexion_right":"elbow_flexion_left",
}

# === 等比例縮放到 224 並回傳縮放與 padding 參數 ===
def letterbox_224_with_params(rgb):
    h, w = rgb.shape[:2]
    s = 224.0 / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s)) 
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC) # 影像等比例縮放
    top = (224 - nh) // 2
    bottom = 224 - nh - top
    left = (224 - nw) // 2
    right = 224 - nw - left
    padded = cv2.copyMakeBorder( # padding成224x224
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, s, left, top  # 224x224, 縮放係數, 左pad, 上pad

# === 2. 資料集類別（整合 ROI Align） ===
def make_cache_path(video_path, do_flip, meta: dict, cache_root="cache_feats"):
    """
    meta 內放會影響特徵的所有超參，避免改參數卻撞舊檔。
    例如: T、roi、scale、backbone、norm、diff、version...
    """
    os.makedirs(cache_root, exist_ok=True)
    # 用「完整路徑 + meta」做 hash，避免重名、確保參數變動就換檔
    key_str = json.dumps(meta, sort_keys=True) + "|" + os.path.abspath(video_path) + f"|flip={int(do_flip)}"
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    fname = f"{stem}-{h}.npz"
    return os.path.join(cache_root, fname)

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform_norm, yolo_model, feature_extractor,
                 max_frames=30, yolo_device='cpu', show=False, is_train=False, flip_prob=0.5):
        self.samples = []  # (PATH, label index)
        self.to_tensor_norm = transform_norm
        self.yolo = yolo_model
        self.feature_extractor = feature_extractor  # MobileNet.features
        self.max_frames = max_frames
        self.yolo_device = yolo_device
        self.show = show
        self.is_train = is_train
        self.flip_prob = float(flip_prob)
        self.detection_log = []

        for label in LABELS:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.mov', '.mp4', '.mkv', '.avi')):
                    self.samples.append((os.path.join(label_dir, fname), label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        do_flip = self.is_train and (np.random.rand() < self.flip_prob)
        if do_flip:
            lbl_name = LABELS[int(label)]
            if lbl_name in FLIP_MAP:
                label = label2idx[FLIP_MAP[lbl_name]]  # 左右互換

        meta = {
            "T": self.max_frames,
            "roi": "align_1x1",
            "backbone": "mbv3_large_features",
            "norm": "imagenet",
            "diff": False,          # 若後面你加了一階差分就改 True
            "version": "v2"         # 你改管線時記得 bump
        }
        cache_path = make_cache_path(video_path, do_flip, meta, cache_root=os.path.join("data","video_features"))


        # 讀快取（安全做法）
        try:
            with np.load(cache_path, allow_pickle=False, mmap_mode="r") as z:
                arr = z["feat"]
                # 基本驗證
                if arr.ndim == 2 and arr.shape[1] in (960, 1920):   # 視你是否拼上差分
                    return torch.from_numpy(arr.astype(np.float32, copy=False)), label
        except Exception:
            pass  # 檔案不存在或壞掉就走計算路徑

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros((self.max_frames, 960), dtype=torch.float32), int(label)

        T = self.max_frames
        feats_ts = []  # list of (ts_ms, feat[960])
        person_detected_frames = 0
        total_frames_read = 0

        # 取得 FPS（可能為 0）
        fps = cap.get(cv2.CAP_PROP_FPS)
        use_fps_ts = fps is not None and fps > 1e-3
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames_read += 1

                # ---- 若此段需要翻轉，對每幀做水平翻轉 ----
                if do_flip:
                    frame = cv2.flip(frame, 1)
                # --- 每幀的時間戳（毫秒） ---
                if use_fps_ts:
                    ts_ms = (frame_idx * 1000.0) / float(fps)
                else:
                    # 有些編碼器在 read() 後取 CAP_PROP_POS_MSEC 也行
                    ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_idx += 1

                # --- YOLO 單幀偵測 ---
                results = self.yolo(frame, device=self.yolo_device, verbose=False)
                if not results or len(results) == 0:
                    continue
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None or boxes.cls is None or boxes.xyxy is None:
                    continue

                cls = boxes.cls.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()

                # 找第一個 person
                got_person = False
                for i in range(len(cls)):
                    if int(cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy[i])

                    # --- 整幀 letterbox 到 224 ---
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img224, scale224, pad_left, pad_top = letterbox_224_with_params(frame_rgb)

                    # 人框座標，後續ROI Align使用
                    x1_224 = x1 * scale224 + pad_left
                    y1_224 = y1 * scale224 + pad_top
                    x2_224 = x2 * scale224 + pad_left
                    y2_224 = y2 * scale224 + pad_top

                    # --- 特徵圖 ---
                    img_tensor = self.to_tensor_norm(img224).unsqueeze(0).to(
                        next(self.feature_extractor.parameters()).device
                    )  # [1,3,224,224]
                    with torch.no_grad():
                        fmap = self.feature_extractor(img_tensor)  # [1,960,7,7]

                    # --- ROI Align（對齊到固定 7x7） ---
                    spatial_scale = fmap.shape[-1] / 224.0  # 7/224
                    boxes_224 = torch.tensor([[x1_224, y1_224, x2_224, y2_224]],
                                            dtype=torch.float32, device=fmap.device)
                    pooled = roi_align(
                        fmap, [boxes_224],
                        output_size=(7, 7),
                        spatial_scale=spatial_scale,
                        sampling_ratio=-1,
                        aligned=True
                    )  # [1,960,7,7]

                    # --- 全域平均池化(GAP) -> 960 維 ---
                    feat = F.adaptive_avg_pool2d(pooled, output_size=1).squeeze(0).squeeze(-1).squeeze(-1)  # [960]
                    feats_ts.append((ts_ms, feat.cpu()))
                    person_detected_frames += 1
                    got_person = True

                    if self.show:
                        vis = img224.copy()
                        cv2.rectangle(
                            vis,
                            (int(round(x1_224)), int(round(y1_224))),
                            (int(round(x2_224)), int(round(y2_224))),
                            (0, 255, 0), 2
                        )
                        cv2.imshow("YOLO ROIAlign (letterboxed 224)", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    break  # 只取第一個人
                # 若此幀沒抓到人，就不 push feats_ts；之後分段池化會做最近鄰補
        finally:
            cap.release()
            if self.show:
                cv2.destroyAllWindows()

        # --- 若整支影片完全沒有特徵，回傳全零 ---
        if len(feats_ts) == 0:
            video_feature = torch.zeros((T, 960), dtype=torch.float32)
            detection_ratio = 0.0
            self.detection_log.append({
                "video": os.path.basename(video_path),
                "detected_frames": 0,
                "total_frames": int(total_frames_read),
                "detection_ratio": float(detection_ratio)
            })
            return video_feature, int(label)

        # === 按時間戳均勻化 + 分段池化（TSN-style）===
        # 在 [t0, t1] 上切成 T 段，段內對多幀做平均；空段用最近鄰補
        ts_arr = np.array([t for t, _ in feats_ts], dtype=np.float64)
        feats_list = [f for _, f in feats_ts]
        t0, t1 = float(ts_arr[0]), float(ts_arr[-1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            # 退化情況：所有 ts 一樣或異常 -> 直接均勻複製最近鄰
            pooled = [feats_list[min(i, len(feats_list)-1)] for i in range(T)]
            video_feature = torch.stack(pooled, dim=0).float()
        else:
            edges = np.linspace(t0, t1, num=T+1, dtype=np.float64)  # T 段的邊界
            pooled_feats = []
            for i in range(T):
                l = edges[i]
                r = edges[i+1]
                # 左閉右開，最後一段右端點包含
                if i < T-1:
                    mask = (ts_arr >= l) & (ts_arr < r)
                else:
                    mask = (ts_arr >= l) & (ts_arr <= r)
                idxs = np.where(mask)[0]

                if idxs.size > 0: # 該區段有幀
                    chunk = torch.stack([feats_list[j] for j in idxs], dim=0)  # [K,960]
                    pooled = chunk.mean(dim=0)
                else: # 該區段無幀
                    # 最近鄰補：取該段中點的最近時間
                    mid = 0.5 * (l + r)
                    j = np.searchsorted(ts_arr, mid)
                    if j <= 0:
                        pooled = feats_list[0]
                    elif j >= len(feats_list):
                        pooled = feats_list[-1]
                    else:
                        pooled = feats_list[j-1] if (mid - ts_arr[j-1]) <= (ts_arr[j] - mid) else feats_list[j]
                pooled_feats.append(pooled)
            video_feature = torch.stack(pooled_feats, dim=0).float()  # [T,960]

        detection_ratio = (person_detected_frames / total_frames_read) if total_frames_read > 0 else 0.0
        self.detection_log.append({
            "video": os.path.basename(video_path),
            "detected_frames": int(person_detected_frames),
            "total_frames": int(total_frames_read),
            "detection_ratio": float(detection_ratio)
        })
        # 否則照原流程抽 -> 存 cache
        np.savez_compressed(cache_path, feat=video_feature.numpy().astype(np.float32, copy=False))
        return video_feature, int(label)
    
    # def export_detection_log(self, csv_path='yolo_detection_ratio.csv'):
    #     with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    #         writer = csv.DictWriter(f, fieldnames=["video", "detected_frames", "total_frames", "detection_ratio"])
    #         writer.writeheader()
    #         for row in self.detection_log:
    #             writer.writerow(row)

# === 3. MobileNet 特徵擷取（取 features 層，供 ROI Align） ===
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobilenet.eval()
feature_extractor = mobilenet.features  # [N,960,7,7]
feature_extractor.eval()

# 抽特徵裝置（可用 GPU 就上；YOLO 先用 CPU 避開 NMS 問題）
extractor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(extractor_device)

# === 4. Normalize（整幀 224×224 使用） ===
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. YOLOv11 模型 ===
yolo = YOLO('yolo11n.pt')
yolo_device = 'cuda'

# === 6. 分割資料 ===
train_dataset = VideoDataset(
    root_dir='data',
    transform_norm=transform_norm,
    yolo_model=yolo,
    feature_extractor=feature_extractor,
    max_frames=30,
    yolo_device=yolo_device,
    show=False,  # 訓練時建議關閉
    is_train=True,
    flip_prob=0.5
)
val_dataset = VideoDataset(
    root_dir='data',
    transform_norm=transform_norm,
    yolo_model=yolo,
    feature_extractor=feature_extractor,
    max_frames=30,
    yolo_device=yolo_device,
    show=False,
    is_train=False,
    flip_prob=0.0
)

video_paths = [p for p, _ in train_dataset.samples]
labels = [l for _, l in train_dataset.samples]

train_idx, val_idx = train_test_split(
    range(len(video_paths)),
    test_size=0.2,
    stratify=labels,  # 分層抽樣，保持類別比例一致
    random_state=42
)

train_set = torch.utils.data.Subset(train_dataset, train_idx)
val_set = torch.utils.data.Subset(val_dataset, val_idx)

# Windows 先用 num_workers=0 跑穩再調
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)  # [B,30,960]
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

# === 7. LSTM 分類模型 ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=960, hidden_size=256, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: [1,B,H]
        return self.fc(hn.squeeze(0))  # [B,9]

model = LSTMClassifier()

# === 8. 損失與優化器 ===
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# === 9. 訓練 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_acc = 0
patience = 10
patience_counter = 0

EPOCHS = 128
for epoch in tqdm(range(EPOCHS), desc="Total Epochs"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        feats = feats.to(device)
        labels = labels.to(device).long()

        preds = model(feats)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100 if total > 0 else 0.0

    # === 驗證階段 ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            feats = feats.to(device)
            labels = labels.to(device).long()
            preds = model(feats)
            val_loss += criterion(preds, labels).item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total * 100 if total > 0 else 0.0

    print(f"Epoch {epoch+1}: TrainLoss={total_loss:.4f} | ValLoss={val_loss:.4f} | "
          f"TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}%")

    # === Early Stopping ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        print(val_acc, best_val_acc, patience_counter)
        torch.save(model.state_dict(), "best_lstm_model_flip.pt")
    else:
        patience_counter += 1
        print(val_acc, best_val_acc, patience_counter)
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# dataset.export_detection_log("yolo_detection_ratio.csv")
# print("YOLO 偵測比例已輸出到 yolo_detection_ratio.csv")

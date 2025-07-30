from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import torch
import torchvision.models as models
import torchvision.transforms as transforms

model = YOLO('yolo11n.pt')

# --- 初始化 CNN (MobileNetV2 特徵提取) ---
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = torch.nn.Identity()  # 移除分類層，只輸出特徵
mobilenet.eval()

# --- 影像前處理 ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 每個人對應的特徵序列 ---
track_features = defaultdict(lambda: deque(maxlen=30))  # 每個人最多 30 幀特徵

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv11 Tracking
    results = model.track(frame, persist=True, verbose=False)
    if not results or len(results) == 0:
        cv2.imshow("YOLOv11 Tracking + CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # results[0] 是當前幀的檢測結果
    boxes = results[0].boxes if hasattr(results[0], 'boxes') else []

    for box in boxes:
        cls = int(box.cls[0])
        track_id = int(box.id[0]) if box.id is not None else -1

        if cls == 0 and track_id != -1:  # 只抓 person 類別
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 裁剪人物圖像
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size > 0:
                # BGR -> RGB
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                # CNN 特徵提取
                img_tensor = transform(person_crop_rgb).unsqueeze(0)  # [1, 3, 224, 224]
                with torch.no_grad():
                    feature = mobilenet(img_tensor).squeeze()  # Tensor shape [1280]
                track_features[track_id].append(feature.cpu())

    cv2.imshow("YOLOv11 Tracking + CNN", frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

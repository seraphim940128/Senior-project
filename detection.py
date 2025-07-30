from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fixed_size = (224, 224)
frame_features_queue = deque(maxlen=30)  # 預留給 LSTM 用的連續幀特徵 (30幀)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 偵測
    results = model(frame, verbose=False)

    person_crops = []  # 存放裁剪後的人框
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person 類別
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 裁剪並 resize
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    resized_person = cv2.resize(person_crop, fixed_size)
                    # 轉成 CNN 輸入格式 (Tensor 格式會放在後續 CNN 處理)
                    person_crops.append(resized_person)

    cv2.imshow("YOLOv11 Real-time Detection", frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

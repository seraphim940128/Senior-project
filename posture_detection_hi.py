import cv2
import numpy as np
import mediapipe as mp
import time

def get_angle(a, b, c):
    a = (a[0]*w, a[1]*h, a[2])
    b = (b[0]*w, b[1]*h, b[2])
    c = (c[0]*w, c[1]*h, c[2])

    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def get_distance(a, b):
    a = (a[0]*w, a[1]*h)
    b = (b[0]*w, b[1]*h)
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def side_tap_left(left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, step=-1, round=0, start=False, start_time=0):

    current_time = time.time()
    
    # 髖寬
    hip_width = get_distance(left_hip, right_hip)
    # 左右腳距離
    left_foot_to_right_foot = get_distance(left_ankle, right_ankle)
    # 左腳到中點距離
    left_foot_to_center = abs(left_ankle[0] - (left_hip[0] + right_hip[0])/2) * w

    left_knee_angle = get_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle)
    
    # 腳沒伸直
    if not (150 < left_knee_angle < 210 and 150 < right_knee_angle < 210):
        cv2.putText(img, "Keep legs straight", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time
    
    # Step -1: 起始
    if step == -1 and left_foot_to_right_foot < hip_width * 1.5:
        step, start_time, start = 0, current_time, True
        cv2.putText(img, "Start Left Side Tap", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("開始左側踢腿")
    
    # Step 0: 向左移動左腿
    elif step == 0 and left_foot_to_center > hip_width * 2.4 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
        else:
            cv2.putText(img, "Move left leg to side slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢向左移動左腿")
    
    # Step 1: 左腿移回中點
    elif step == 1 and left_foot_to_right_foot < hip_width * 1.5 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
            cv2.putText(img, f"Completed Left Side Tap {round}", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"完成左側踢腿第 {round} 次！")
        else:
            cv2.putText(img, "Return leg to center slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢將腿收回中間")
    
    return step, round, start, start_time

def side_tap_right(left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, step=-1, round=0, start=False, start_time=0):
    # 功能等同side_tap_left
    current_time = time.time()
    
    hip_width = get_distance(left_hip, right_hip)
    left_foot_to_right_foot = get_distance(left_ankle, right_ankle)
    right_foot_to_center = abs(right_ankle[0] - (left_hip[0] + right_hip[0])/2) * w
    
    left_knee_angle = get_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle)
    
    if not (150 < left_knee_angle < 210 and 150 < right_knee_angle < 210):
        cv2.putText(img, "Keep legs straight", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time

    if step == -1 and left_foot_to_right_foot < hip_width * 1.5:
        step, start_time, start = 0, current_time, True
        cv2.putText(img, "Start Right Side Tap", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("開始右側踢腿")
    
    elif step == 0 and right_foot_to_center > hip_width * 2.4 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
        else:
            cv2.putText(img, "Move right leg to side slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢向右移動右腿")
    
    elif step == 1 and left_foot_to_right_foot < hip_width * 1.5 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
            cv2.putText(img, f"Completed Right Side Tap {round}", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"完成右側踢腿第 {round} 次！")
        else:
            cv2.putText(img, "Return leg to center slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢將腿收回中間")
    
    return step, round, start, start_time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("無法開啟相機")
        exit()
    
    # 初始化參數
    left_side_tap_step = -1
    right_side_tap_step = -1
    left_side_tap_round = 0
    right_side_tap_round = 0
    left_side_tap_start = False
    right_side_tap_start = False
    left_side_tap_start_time = 0
    right_side_tap_start_time = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            h, w, _ = img.shape
            landmarks = results.pose_landmarks.landmark

            pose_dict = {}
            for i, lm in enumerate(landmarks):
                name = mp_pose.PoseLandmark(i).name
                cx, cy, cz = lm.x, lm.y, lm.z
                globals()[name] = (cx, cy, cz)
                pose_dict[name] = (cx, cy, cz)

            # Left side tap 偵測
            left_side_tap_step, left_side_tap_round, left_side_tap_start, left_side_tap_start_time = \
                side_tap_left(pose_dict["LEFT_HIP"], pose_dict["LEFT_KNEE"], pose_dict["LEFT_ANKLE"],
                            pose_dict["RIGHT_HIP"], pose_dict["RIGHT_KNEE"], pose_dict["RIGHT_ANKLE"],
                            left_side_tap_step, left_side_tap_round, left_side_tap_start, left_side_tap_start_time)
            
            # Left side tap 顯示
            cv2.putText(img, f'Left Side Tap:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Step: {left_side_tap_step}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Round: {left_side_tap_round}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 等同左腳到中點距離
            cv2.putText(img, f'Left Foot Distance: {abs(pose_dict["LEFT_ANKLE"][0] - (pose_dict["LEFT_HIP"][0] + pose_dict["RIGHT_HIP"][0])/2) * w:.1f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Right side tap 偵測
            right_side_tap_step, right_side_tap_round, right_side_tap_start, right_side_tap_start_time = \
                side_tap_right(pose_dict["LEFT_HIP"], pose_dict["LEFT_KNEE"], pose_dict["LEFT_ANKLE"],
                             pose_dict["RIGHT_HIP"], pose_dict["RIGHT_KNEE"], pose_dict["RIGHT_ANKLE"],
                             right_side_tap_step, right_side_tap_round, right_side_tap_start, right_side_tap_start_time)
            
            # Right side tap 顯示
            cv2.putText(img, f'Right Side Tap:', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img, f'Step: {right_side_tap_step}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img, f'Round: {right_side_tap_round}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img, f'Right Foot Distance: {abs(pose_dict["RIGHT_ANKLE"][0] - (pose_dict["LEFT_HIP"][0] + pose_dict["RIGHT_HIP"][0])/2) * w:.1f}', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        #放大方便看
        cv2.namedWindow('Side Tap Detection', cv2.WINDOW_NORMAL)

        cv2.imshow('Side Tap Detection', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
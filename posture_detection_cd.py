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

def is_arm_straight(shoulder, elbow, wrist, tolerance=0.1):
    a = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    b = np.linalg.norm(np.array(elbow) - np.array(wrist))
    c = np.linalg.norm(np.array(shoulder) - np.array(wrist))
    return abs((a + b) - c) < tolerance

def shoulder_flexion(shoulder, elbow, wrist, hip, step=-1, round=0, start=False, start_time=0):
    if not is_arm_straight(shoulder, elbow, wrist):
        print("請伸直手臂")
        cv2.putText(img, "Please stretch your arms", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time

    max_horizontal_diff = 0.2  
    horizontal_diff = abs(wrist[0] - shoulder[0])
    if horizontal_diff > max_horizontal_diff:
        print("請直舉而非側舉")
        cv2.putText(img, "Not a shoulder flexion", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time

    angle_shoulder = get_angle(wrist, shoulder, hip)
    current_time = time.time()

    if step == -1 and 10 < angle_shoulder < 30:
        step, start_time, start = 0, current_time, True
        cv2.putText(img, "Start Action", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("開始動作")

    elif step == 0 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
        else:
            cv2.putText(img, "Please raise your arms slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢抬起手")

    elif step == 1 and 160 < angle_shoulder < 190 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time
        else:
            cv2.putText(img, "Keep raising your arms", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請繼續舉高手臂")

    elif step == 2 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time
        else:
            cv2.putText(img, "Lower your arms slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢放下手臂")

    elif step == 3 and 10 < angle_shoulder < 30 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            cv2.putText(img, f"Completed round {round}", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"完成第 {round} 次！")
        else:
            cv2.putText(img, "Keep lowering your arms", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢放下手臂")

    return step, round, start, start_time



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if not cap.isOpened():
        print("無法開啟相機")
        exit()

    left_shoulder_flexion_step, right_shoulder_flexion_step = -1, -1
    left_shoulder_flexion_round, right_shoulder_flexion_round = 0, 0
    left_shoulder_flexion_start, right_shoulder_flexion_start = False, False
    left_shoulder_flexion_time, right_shoulder_flexion_time = 0, 0

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
                pose_dict[name] = (lm.x, lm.y, lm.z)

            # 左手
            left_shoulder_flexion_step, left_shoulder_flexion_round, left_shoulder_flexion_start, left_shoulder_flexion_time = shoulder_flexion(
                pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"], pose_dict["LEFT_HIP"],
                left_shoulder_flexion_step, left_shoulder_flexion_round, left_shoulder_flexion_start, left_shoulder_flexion_time)
            cv2.putText(img, f'Left shoulder flexion:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Step: {left_shoulder_flexion_step}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Round: {left_shoulder_flexion_round}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            angle_l = get_angle(pose_dict["LEFT_WRIST"], pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_HIP"])
            cv2.putText(img, f'Angle: {int(angle_l)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 右手
            right_shoulder_flexion_step, right_shoulder_flexion_round, right_shoulder_flexion_start, right_shoulder_flexion_time = shoulder_flexion(
                pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"], pose_dict["RIGHT_HIP"],
                right_shoulder_flexion_step, right_shoulder_flexion_round, right_shoulder_flexion_start, right_shoulder_flexion_time)
            cv2.putText(img, f'Right shoulder flexion:', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Step: {right_shoulder_flexion_step}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Round: {right_shoulder_flexion_round}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            angle_r = get_angle(pose_dict["RIGHT_WRIST"], pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_HIP"])
            cv2.putText(img, f'Angle: {int(angle_r)}', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Shoulder Flexion Detection', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import time

VIS_THR = 0.6   # 0~1 之間，可見度門檻

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

def elbow_flexion(shoulder, elbow, wrist, step=-1, round=0, start=False, start_time=0):
    max_upper_arm_shift = 0.15
    if abs(elbow[0] - shoulder[0]) > max_upper_arm_shift:
        cv2.putText(img, "Keep upper arm still", (int(w/3), h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print("請固定上臂，避免前後擺動")
        return step, round, start, start_time

    angle_elbow   = get_angle(shoulder, elbow, wrist)
    current_time  = time.time()

    if step == -1 and 160 < angle_elbow < 185:
        step, start_time, start = 0, current_time, True
        cv2.putText(img, "Start Action", (int(w/3), h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("開始動作 (手臂伸直)")

    elif step == 0 and 40 < angle_elbow < 80 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
        else:
            cv2.putText(img, "Keep bending your elbow", (int(w/3), h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請持續彎曲手肘")

    elif step == 1 and 160 < angle_elbow < 185 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            cv2.putText(img, f"Completed round {round}", (int(w/3), h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"完成第 {round} 次！")
        else:
            cv2.putText(img, "Straighten your arm", (int(w/3), h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請完全伸直手臂")

    return step, round, start, start_time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if not cap.isOpened():
        print("無法開啟相機")
        exit()

    left_elbow_flexion_step, right_elbow_flexion_step = -1, -1
    left_elbow_flexion_round, right_elbow_flexion_round = 0, 0
    left_elbow_flexion_start, right_elbow_flexion_start = False, False
    left_elbow_flexion_time, right_elbow_flexion_time = 0, 0

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

            left_visible  = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility  > VIS_THR and
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value   ].visibility   > VIS_THR and
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value   ].visibility   > VIS_THR)

            right_visible = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > VIS_THR and
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value   ].visibility  > VIS_THR and
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value   ].visibility  > VIS_THR)

            # 左手
            if left_visible:
                left_elbow_flexion_step, left_elbow_flexion_round, left_elbow_flexion_start, left_elbow_flexion_time = elbow_flexion(
                    pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"],
                    left_elbow_flexion_step, left_elbow_flexion_round, left_elbow_flexion_start, left_elbow_flexion_time)
                cv2.putText(img, f'Left elbow flexion:', (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,  0), 2)
                cv2.putText(img, f'Step:  {left_elbow_flexion_step}',  (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,  0), 2)
                cv2.putText(img, f'Round: {left_elbow_flexion_round}', (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,  0), 2)
                angle_l = get_angle(pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"])
                cv2.putText(img, f'Angle: {int(angle_l)}',            (10,120),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,  0), 2)
            else:
                cv2.putText(img, "Put LEFT arm in view", (int(w/3), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                angle_l = 0

            # 右手
            if right_visible:
                right_elbow_flexion_step, right_elbow_flexion_round, right_elbow_flexion_start, right_elbow_flexion_time = elbow_flexion(
                    pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"],
                    right_elbow_flexion_step, right_elbow_flexion_round, right_elbow_flexion_start, right_elbow_flexion_time)
                cv2.putText(img, f'Right elbow flexion:', (10,180),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,  0,  0), 2)
                cv2.putText(img, f'Step:  {right_elbow_flexion_step}', (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,  0,  0), 2)
                cv2.putText(img, f'Round: {right_elbow_flexion_round}',(10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,  0,  0), 2)
                angle_r = get_angle(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"])
                cv2.putText(img, f'Angle: {int(angle_r)}',            (10,270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,  0,  0), 2)
            else:
                cv2.putText(img, "Put RIGHT arm in view", (int(w/3), 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                angle_r = 0  

        cv2.imshow('Shoulder Flexion Detection', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
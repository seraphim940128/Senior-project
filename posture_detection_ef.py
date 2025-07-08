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

def shoulder_abduction(img, shoulder, elbow, wrist, hip, step=-1, round=0, start=False, start_time=0):
    """
    step: -1: not started, 0: start, 1: raising arms, 2: middle, 3: lowering arms
    round: number of completed rounds
    start: whether the action has started
    start_time: time when the action started
    """
    angle_shoulder = get_angle(hip, shoulder, elbow)
    # angle_elbow = get_angle(shoulder, elbow, wrist)
    current_time = time.time()

    if not is_arm_straight(shoulder, elbow, wrist):
        print("請伸直手臂")
        cv2.putText(img, "Please stretch your arms", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time
    
    # max_horizontal_diff = 0.2  
    # horizontal_diff = abs(wrist[0] - shoulder[0])
    max_depth_diff = 0.07
    # Considering the difference of arm's length for each person, using depth instead of horizontal distance, 
    # hip is used as the reference point to calculate the depth difference
    depth_diff = abs((wrist[2]-hip[2]) - (shoulder[2]-hip[2])) 
    if depth_diff > max_depth_diff:
        # print("請側舉而非直舉: ", horizontal_diff, " ", depth_diff, " ", wrist[2], " ", shoulder[2])
        cv2.putText(img, "Not a shoulder abduction", (int(w/3), h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # step, start, start_time = -1, False, 0
        return step, round, start, start_time
    # print(horizontal_diff, " ", depth_diff, " ", wrist[2], " ", shoulder[2])

    if step == -1 and 15 < angle_shoulder < 25:
        step, start_time, start = 0, current_time, True
        cv2.putText(img, "Start Action", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("開始動作")

    elif step == 0 and 85 < angle_shoulder < 95 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
        else:
            cv2.putText(img, "Please raise your hands slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢抬起手")

    elif step == 1 and 160 < angle_shoulder < 180 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time
        else:
            cv2.putText(img, "Please raise your hands slowly", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢舉高手")
    
    elif step == 2 and 85 < angle_shoulder < 95 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time
        else:
            cv2.putText(img, "Please lower your arms slowly.", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢放下手臂")

    elif step == 3 and 5 < angle_shoulder < 20 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            cv2.putText(img, f"Completed round {round}", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"完成第 {round} 次！")
        else:
            cv2.putText(img, "Please lower your arms slowly.", (int(w/3), h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("請慢慢放下手臂")

    return step, round, start, start_time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("無法開啟相機")
        exit()
    
    # 初始化肩部外展的計數器和狀態
    left_shoulder_abduction_step = -1
    right_shoulder_abduction_step = -1
    left_shoulder_abduction_round = 0
    right_shoulder_abduction_round = 0
    left_shoulder_abduction_start = False
    right_shoulder_abduction_start = False
    left_shoulder_abduction_start_time = 0
    right_shoulder_abduction_start_time = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        # img = cv2.resize(img, (520, 300))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = img.shape
            landmarks = results.pose_landmarks.landmark
            # print(img.shape)
            pose_dict = {}
            for i, lm in enumerate(landmarks):
                name = mp_pose.PoseLandmark(i).name
                cx, cy, cz = lm.x, lm.y, lm.z
                globals()[name] = (cx, cy, cz)
                pose_dict[name] = (cx, cy, cz)  

            #X:[0] Y:[1] Z:[2] 
            
            left_shoulder_abduction_step, left_shoulder_abduction_round, left_shoulder_abduction_start, left_shoulder_abduction_start_time = \
                shoulder_abduction(img, pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"], pose_dict["LEFT_HIP"],
                left_shoulder_abduction_step, left_shoulder_abduction_round, left_shoulder_abduction_start, left_shoulder_abduction_start_time)
            cv2.putText(img, f'Left shoulder abduction:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Step: {left_shoulder_abduction_step}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Round: {left_shoulder_abduction_round}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Angle: {get_angle(pose_dict["LEFT_HIP"], pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"])}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Angle(180): {get_angle(pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"])}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            right_shoulder_abduction_step, right_shoulder_abduction_round, right_shoulder_abduction_start, right_shoulder_abduction_start_time = \
                shoulder_abduction(img, pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"], pose_dict["RIGHT_HIP"],
                right_shoulder_abduction_step, right_shoulder_abduction_round, right_shoulder_abduction_start, right_shoulder_abduction_start_time)
            cv2.putText(img, f'Right shoulder abduction:', (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Step: {right_shoulder_abduction_step}', (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Round: {right_shoulder_abduction_round}', (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Angle: {get_angle(pose_dict["RIGHT_HIP"], pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"])}', (0, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'Angle(180): {get_angle(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"])}', (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
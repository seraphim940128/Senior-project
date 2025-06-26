import cv2
import numpy as np
import mediapipe as mp

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

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法讀取影像")
            break

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

            pose_dict = {}
            for i, lm in enumerate(landmarks):
                name = mp_pose.PoseLandmark(i).name
                cx, cy, cz = lm.x, lm.y, lm.z
                globals()[name] = (cx, cy, cz)
                pose_dict[name] = (cx, cy, cz)  

            def calculate_angle(a, b, c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                return np.degrees(angle)

            def is_arm_raised(shoulder, wrist, threshold=0.05):
                return wrist[1] < shoulder[1] - threshold
            
            def is_oncoming_traffic_right(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05):
                upper = shoulder_l[1] + threshold
                lower = shoulder_l[1] - threshold
                return (
                    lower < wrist_l[1] < upper and lower < wrist_r[1] < upper and
                    lower < shoulder_l[1] < upper and lower < shoulder_r[1] < upper and
                    lower < elbow_l[1] < upper and lower < elbow_r[1] < upper and
                    wrist_l[0] > shoulder_l[0] and wrist_r[0] > elbow_r[0]
                )

            def is_oncoming_traffic_left(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05):
                upper = shoulder_l[1] + threshold
                lower = shoulder_l[1] - threshold
                cv2.line(img, (0, int(upper * h)), (w, int(upper * h)), (255, 0, 0), 2)
                cv2.line(img, (0, int(lower * h)), (w, int(lower * h)), (0, 255, 0), 2)
                return (
                    wrist_l[1] < upper and
                    lower < wrist_r[1] < upper and
                    lower < shoulder_l[1] < upper and
                    lower < shoulder_r[1] < upper and
                    lower < elbow_l[1] < upper and
                    lower < elbow_r[1] < upper and
                    wrist_l[0] < elbow_l[0] and wrist_r[0] < elbow_r[0]
                )

            def is_right_car_turns_left(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05, threshold_angle=15):
                upper = shoulder_l[1] + threshold
                lower = shoulder_l[1] - threshold
                left_angle = calculate_angle((shoulder_l[0]*w, shoulder_l[1]*h, shoulder_l[2]), (elbow_l[0]*w, elbow_l[1]*h, elbow_l[2]), (wrist_l[0]*w, wrist_l[1]*h, wrist_l[2]))
                right_angle = calculate_angle((shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]), (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]), (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2]))
                return (
                    lower < shoulder_l[1] < upper and lower < shoulder_r[1] < upper and
                    lower < elbow_l[1] < upper and lower < elbow_r[1] < upper and
                    abs(left_angle - 90) < threshold_angle and
                    abs(right_angle - 35) < threshold_angle and
                    wrist_l[1] < lower and wrist_r[1] < lower
                )

            def is_left_car_turns_left(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05):
                upper = shoulder_l[1] + threshold
                lower = shoulder_l[1] - threshold
                left_angle = calculate_angle((shoulder_l[0]*w, shoulder_l[1]*h, shoulder_l[2]), (elbow_l[0]*w, elbow_l[1]*h, elbow_l[2]), (wrist_l[0]*w, wrist_l[1]*h, wrist_l[2]))
                right_angle = calculate_angle((shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]), (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]), (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2]))
                return (
                    lower < shoulder_l[1] < upper and lower < shoulder_r[1] < upper and
                    lower < elbow_l[1] < upper and lower < elbow_r[1] < upper and
                    abs(left_angle) < 10 and abs(right_angle - 90) < 10 and
                    wrist_r[1] < lower
                )

            def is_stop_all(shoulder_r, elbow_r, wrist_r, threshold_y=0.05, angle_threshold=160):
                if wrist_r[1] < shoulder_r[1] - threshold_y:
                    angle = calculate_angle(
                        (shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]),
                        (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]),
                        (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2])
                    )
                    if angle > angle_threshold:
                        return True
                return False
            
            def is_forward_stop_gesture(shoulder_r, elbow_r, wrist_r, angle_target=90, angle_threshold=15):
                angle = calculate_angle(
                    (shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]),
                    (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]),
                    (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2])
                )
                
                is_upper_arm_horizontal = abs(elbow_r[1] - shoulder_r[1]) < 0.05
                
                is_forearm_upward = wrist_r[1] < elbow_r[1]
                
                if abs(angle - angle_target) < angle_threshold and is_upper_arm_horizontal and is_forearm_upward:
                    return True
                return False
            

            left_raised = is_arm_raised(pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_WRIST"])
            right_raised = is_arm_raised(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_WRIST"])
            oncoming_traffic_right = is_oncoming_traffic_right(
                pose_dict["LEFT_SHOULDER"], pose_dict["RIGHT_SHOULDER"],
                pose_dict["LEFT_WRIST"], pose_dict["RIGHT_WRIST"],
                pose_dict["LEFT_ELBOW"], pose_dict["RIGHT_ELBOW"]
            )
            oncoming_traffic_left = is_oncoming_traffic_left(
                pose_dict["LEFT_SHOULDER"], pose_dict["RIGHT_SHOULDER"],
                pose_dict["LEFT_WRIST"], pose_dict["RIGHT_WRIST"],
                pose_dict["LEFT_ELBOW"], pose_dict["RIGHT_ELBOW"]
            )
            right_turns_left = is_right_car_turns_left(
                pose_dict["LEFT_SHOULDER"], pose_dict["RIGHT_SHOULDER"],
                pose_dict["LEFT_WRIST"], pose_dict["RIGHT_WRIST"],
                pose_dict["LEFT_ELBOW"], pose_dict["RIGHT_ELBOW"]
            )
            left_turns_left = is_left_car_turns_left(
                pose_dict["LEFT_SHOULDER"], pose_dict["RIGHT_SHOULDER"],
                pose_dict["LEFT_WRIST"], pose_dict["RIGHT_WRIST"],
                pose_dict["LEFT_ELBOW"], pose_dict["RIGHT_ELBOW"]
            )
            stop_all = is_stop_all(
                pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"]
            )
            forward_stop = is_forward_stop_gesture(
                pose_dict["RIGHT_SHOULDER"],
                pose_dict["RIGHT_ELBOW"],
                pose_dict["RIGHT_WRIST"]
            )
            


            if oncoming_traffic_right:
                cv2.putText(img, 'Oncoming traffic right', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            elif oncoming_traffic_left:
                cv2.putText(img, 'Oncoming traffic left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            elif right_turns_left:
                cv2.putText(img, 'Right car turns left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            elif left_turns_left:
                cv2.putText(img, 'Left car turns left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            elif stop_all:
                cv2.putText(img, 'STOP ALL VEHICLES', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            elif forward_stop:
                cv2.putText(img, 'FORWARD TRAFFIC STOP', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            


        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

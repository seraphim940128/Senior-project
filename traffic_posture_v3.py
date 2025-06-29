import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("無法開啟相機")
        exit()

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def is_oncoming_traffic_right(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05):
        """
        右方來車速行
        """
        upper = shoulder_l[1] + threshold
        lower = shoulder_l[1] - threshold
        return (
            lower < wrist_l[1] < upper and lower < wrist_r[1] < upper and
            lower < shoulder_l[1] < upper and lower < shoulder_r[1] < upper and
            lower < elbow_l[1] < upper and lower < elbow_r[1] < upper and
            wrist_l[0] > shoulder_l[0] and wrist_r[0] > elbow_r[0]
        )

    def is_oncoming_traffic_left(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l, elbow_r, threshold=0.05):
        """
        左方來車速行
        """
        upper = shoulder_l[1] + threshold
        lower = shoulder_l[1] - threshold
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
        """
        右方來車左轉
        """
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
        """
        左方來車左轉
        """
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
        """
        停止所有車輛
        """
        if wrist_r[1] < shoulder_r[1] - threshold_y:
            angle = calculate_angle(
                (shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]),
                (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]),
                (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2])
            )
            return angle > angle_threshold
        return False

    def is_forward_stop_gesture(shoulder_r, elbow_r, wrist_r, angle_target=90, angle_threshold=15):
        """
        前方來車停止
        """
        angle = calculate_angle(
            (shoulder_r[0]*w, shoulder_r[1]*h, shoulder_r[2]),
            (elbow_r[0]*w, elbow_r[1]*h, elbow_r[2]),
            (wrist_r[0]*w, wrist_r[1]*h, wrist_r[2])
        )
        is_upper_arm_horizontal = abs(elbow_r[1] - shoulder_r[1]) < 0.05
        is_forearm_upward = wrist_r[1] < elbow_r[1]
        return abs(angle - angle_target) < angle_threshold and is_upper_arm_horizontal and is_forearm_upward

    def is_palm_forward(hand_landmarks):
        """
        判斷手掌是否面向前方
        """
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        middle_finger = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
        return middle_finger[2] < wrist[2]

    def is_left_upper_arm_horizontal(shoulder, elbow, threshold_angle=15):
        """
        判斷左上臂是否水平
        """
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return angle < threshold_angle or angle > (180 - threshold_angle)

    def is_left_forearm_vertical(elbow, wrist, threshold_angle=20):
        """
        判斷左小臂是否垂直
        """
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return abs(angle - 90) < threshold_angle

    def is_left_palm_facing_left(hand_landmarks):
        """
        判斷左手掌是否面向左方
        """
        wrist_x = hand_landmarks.landmark[0].x
        index_mcp_x = hand_landmarks.landmark[5].x
        middle_mcp_x = hand_landmarks.landmark[9].x
        return wrist_x > index_mcp_x and wrist_x > middle_mcp_x

    def is_right_upper_arm_horizontal(shoulder, elbow, threshold_angle=15):
        """
        判斷右上臂是否水平
        """
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return angle < threshold_angle or angle > (180 - threshold_angle)

    def is_right_forearm_vertical(elbow, wrist, threshold_angle=20):
        """
        判斷右小臂是否垂直
        """
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return abs(angle - 90) < threshold_angle

    def is_right_palm_facing_right(hand_landmarks):
        """ 
        判斷右手掌是否面向右方
        """
        wrist_x = hand_landmarks.landmark[0].x
        index_mcp_x = hand_landmarks.landmark[5].x
        middle_mcp_x = hand_landmarks.landmark[9].x
        return wrist_x < index_mcp_x and wrist_x < middle_mcp_x

    def is_both_arms_horizontal(shoulder_l, shoulder_r, elbow_l, elbow_r, wrist_l, wrist_r, angle_threshold=20, y_threshold=0.05):
        """
        判斷雙臂是否水平"""
        def is_arm_horizontal(shoulder, elbow, wrist):
            angle = calculate_angle(
                (shoulder[0]*w, shoulder[1]*h, shoulder[2]),
                (elbow[0]*w, elbow[1]*h, elbow[2]),
                (wrist[0]*w, wrist[1]*h, wrist[2])
            )
            y_aligned = abs(elbow[1] - shoulder[1]) < y_threshold
            return angle > (180 - angle_threshold) and y_aligned
        return is_arm_horizontal(shoulder_l, elbow_l, wrist_l) and is_arm_horizontal(shoulder_r, elbow_r, wrist_r)

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        hand_results = hands.process(img_rgb)
        h, w, _ = img.shape

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            landmarks = results.pose_landmarks.landmark
            pose_dict = {
                mp_pose.PoseLandmark(i).name: (lm.x, lm.y, lm.z)
                for i, lm in enumerate(landmarks)
            }

            stop_all = is_stop_all(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"])

            forward_stop = False
            if all(k in pose_dict for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]):
                if is_forward_stop_gesture(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"]):
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            if is_palm_forward(hand_landmarks):
                                forward_stop = True
                                break

            both_arms_horizontal = False
            if all(k in pose_dict for k in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]):
                both_arms_horizontal = is_both_arms_horizontal(
                    pose_dict["LEFT_SHOULDER"], pose_dict["RIGHT_SHOULDER"],
                    pose_dict["LEFT_ELBOW"], pose_dict["RIGHT_ELBOW"],
                    pose_dict["LEFT_WRIST"], pose_dict["RIGHT_WRIST"]
                )

            left_traffic_stop = False
            right_traffic_stop = False
            left_hand_landmarks = None
            right_hand_landmarks = None

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    if hand_handedness.classification[0].label == "Left":
                        left_hand_landmarks = hand_landmarks
                    elif hand_handedness.classification[0].label == "Right":
                        right_hand_landmarks = hand_landmarks

            if all(k in pose_dict for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]) and left_hand_landmarks:
                if is_left_upper_arm_horizontal(pose_dict["LEFT_SHOULDER"], pose_dict["LEFT_ELBOW"]) and \
                   is_left_forearm_vertical(pose_dict["LEFT_ELBOW"], pose_dict["LEFT_WRIST"]) and \
                   is_left_palm_facing_left(left_hand_landmarks):
                    left_traffic_stop = True

            if all(k in pose_dict for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]) and right_hand_landmarks:
                if is_right_upper_arm_horizontal(pose_dict["RIGHT_SHOULDER"], pose_dict["RIGHT_ELBOW"]) and \
                   is_right_forearm_vertical(pose_dict["RIGHT_ELBOW"], pose_dict["RIGHT_WRIST"]) and \
                   is_right_palm_facing_right(right_hand_landmarks):
                    right_traffic_stop = True

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

            if oncoming_traffic_right:
                cv2.putText(img, 'Oncoming traffic right', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif oncoming_traffic_left:
                cv2.putText(img, 'Oncoming traffic left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif right_turns_left:
                cv2.putText(img, 'Right car turns left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif left_turns_left:
                cv2.putText(img, 'Left car turns left', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif both_arms_horizontal:
                cv2.putText(img, 'FRONT AND REAR STOP', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif stop_all:
                cv2.putText(img, 'STOP ALL VEHICLES', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif left_traffic_stop:
                cv2.putText(img, 'LEFT TRAFFIC STOP', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif right_traffic_stop:
                cv2.putText(img, 'RIGHT TRAFFIC STOP', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif forward_stop:
                cv2.putText(img, 'FORWARD TRAFFIC STOP', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

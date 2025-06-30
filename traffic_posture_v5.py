import cv2
import numpy as np
import math
import mediapipe as mp

# === 初始化 MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# === 計算關節角度 ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_angle_to_horizontal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# === 判斷頭部方向 ===
def get_head_direction(pose, threshold=0.03):
    try:
        nose_x = pose["NOSE"][0]
        left_shoulder_x = pose["LEFT_SHOULDER"][0]
        right_shoulder_x = pose["RIGHT_SHOULDER"][0]

        mid_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2

        if nose_x < mid_shoulder_x - threshold:
            return "Looking Right", (0, 255, 0)       # Green
        elif nose_x > mid_shoulder_x + threshold:
            return "Looking Left", (255, 0, 0)        # Blue
        else:
            return "Looking Forward", (255, 255, 255) # White
    except KeyError:
        return "Unknown", (128, 128, 128)

# === 抓取點座標 ===
# def get_point(pose, name):
#     return np.array(pose[name]) if name in pose else None
def get_point(pose, name, landmarks=None, visibility_threshold=0.5):
    if name in pose:
        point = np.array(pose[name])
        if landmarks:
            idx = mp_pose.PoseLandmark[name].value
            if landmarks[idx].visibility < visibility_threshold:
                return None
        return point
    return None

# === 計算手臂方向角度 ===
def angle_between(p1, p2, axis='x'):
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) if axis == 'x' else np.degrees(np.arctan2(p2[0]-p1[0], p2[1]-p1[1]))

# === 判定方向標籤 ===
def get_angle_label(angle, up_range, horizontal_range):
    if up_range[0] <= angle <= up_range[1]:
        return "Raised Up"
    elif horizontal_range[0] <= angle <= horizontal_range[1]:
        return "Extended Horizontally"
    return "Other"

# === 分析手臂與手掌 ===
def analyze_arm_positions(pose, w, h):
    results = []

    # Right Arm
    shoulder_r = get_point(pose, "RIGHT_SHOULDER", landmarks)
    elbow_r = get_point(pose, "RIGHT_ELBOW", landmarks)
    wrist_r = get_point(pose, "RIGHT_WRIST", landmarks)
    if shoulder_r is not None and elbow_r is not None:
        angle_r = angle_between(shoulder_r, elbow_r)
        direction_r = get_angle_label(angle_r, (-135, -45), (-30, 30))
        results.append(f"Right Arm: {direction_r}")
    else:
        results.append("Right Arm: Other")

    # Left Arm
    shoulder_l = get_point(pose, "LEFT_SHOULDER", landmarks)
    elbow_l = get_point(pose, "LEFT_ELBOW", landmarks)
    wrist_l = get_point(pose, "LEFT_WRIST", landmarks)
    if shoulder_l is not None and elbow_l is not None:
        angle_l = angle_between(shoulder_l, elbow_l)
        direction_l = get_angle_label(angle_l, (-135, -45), (-30, 30))
        results.append(f"Left Arm: {direction_l}")
    else:
        results.append("Left Arm: Other")

    # Right Upper Arm
    if shoulder_r is not None and elbow_r is not None:
        dx = abs(elbow_r[0] - shoulder_r[0])
        dy = abs(elbow_r[1] - shoulder_r[1])
        if dx > 0.05 and dy < 0.08:
            direction = "Extended Right"
        else:
            direction = "Other"
        results.append(f"Right Upper Arm: {direction}")
    else:
        results.append("Right Upper Arm: Other")

    # Left Upper Arm 
    if shoulder_l is not None and elbow_l is not None:
        dx = abs(elbow_l[0] - shoulder_l[0])
        dy = abs(elbow_l[1] - shoulder_l[1])
        if dx > 0.05 and dy < 0.08:
            direction = "Extended Left"
        else:
            direction = "Other"
        results.append(f"Left Upper Arm: {direction}")
    else:
        results.append("Left Upper Arm: Other")

    # Right Arm Half-Raised
    if shoulder_r is not None and elbow_r is not None:
        angle = get_angle_to_horizontal(shoulder_r, elbow_r)
        # print(f"[DEBUG] Right arm angle to horizontal: {angle:.2f}")
        if -135 < angle < -115:
            results.append("Right Arm: Half-Raised")
        else:
            results.append("Right Arm: Not Half-Raised")
    else:
        results.append("Right Arm: Half-Raised: Not Visible")

        # Left Arm Half-Raised
    if shoulder_l is not None and elbow_l is not None:
        angle = get_angle_to_horizontal(shoulder_l, elbow_l)
        # print(f"[DEBUG] Left arm angle to horizontal: {angle:.2f}")
        if -135 < angle < -115:
            results.append("Left Arm: Half-Raised")
        else:
            results.append("Left Arm: Not Half-Raised")
    else:
        results.append("Left Arm: Half-Raised: Not Visible")

    # Right Forearm
    if shoulder_r is not None and elbow_r is not None and wrist_r is not None:
        forearm_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
        if forearm_angle_r > 150:
            results.append("Right Forearm: Fully Raised")
        elif forearm_angle_r > 60:
            results.append("Right Forearm: Partially Raised")
        else:
            results.append("Right Forearm: Other")
    else:
        results.append("Right Forearm: Other")

    # Left Forearm
    if shoulder_l is not None and elbow_l is not None and wrist_l is not None:
        forearm_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
        if forearm_angle_l > 150:
            results.append("Left Forearm: Fully Raised")
        elif forearm_angle_l > 60:
            results.append("Left Forearm: Partially Raised")
        else:
            results.append("Left Forearm: Other")
    else:
        results.append("Left Forearm: Other")

    # Right Palm
    right_index = get_point(pose, "RIGHT_INDEX", landmarks)
    if wrist_r is not None and right_index is not None:
        dz = wrist_r[2] - right_index[2]
        if dz > 0.02:
            results.append("Right Palm: Facing Forward")
        else:
            results.append("Right Palm: Other")
    else:
        results.append("Right Palm: Other")


    # Left Palm
    left_index = get_point(pose, "LEFT_INDEX", landmarks)
    if wrist_l is not None and left_index is not None:
        dz = wrist_l[2] - left_index[2]
        if dz > 0.02:
            results.append("Left Palm: Facing Forward")
        else:
            results.append("Left Palm: Other")
    else:
        results.append("Left Palm: Other")

    return results


# === 姿勢比對 ===
def detect_traffic_pose(analysis, head_direction):
    pose_map = {
        "All Stop": [
            "Right Arm: Raised Up",
            "Right Forearm: Fully Raised",
            "Right Palm: Facing Forward"
        ],
        "Front Stop": [
            "Right Upper Arm: Extended Right",
            "Right Forearm: Partially Raised",
            "Right Palm: Facing Forward",
            "Head: Looking Forward",
            "Left Upper Arm: Other",
        ],
        "Right Stop": [
            "Right Upper Arm: Extended Right",
            "Right Forearm: Partially Raised",
            # "Right Palm: Other",
            "Head: Looking Right"
        ],
        "Left Stop": [
            "Left Upper Arm: Extended Left",
            "Left Forearm: Partially Raised",
            # "Left Palm: Other",
            "Head: Looking Left"
        ],
        "Side Go": [
            "Right Upper Arm: Extended Right",
            "Left Upper Arm: Extended Left",
            "Right Palm: Facing Forward",
            "Left Palm: Facing Forward",
            "Head: Looking Forward",
            "Left Arm: Extended Horizontally"
        ]

    }

    matched = []
    all_features = analysis + [f"Head: {head_direction}"]
    for pose_name, keywords in pose_map.items():
        if all(k in all_features for k in keywords):
            matched.append(pose_name)

    return matched

# === main loop ===
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_tracker:
    if not cap.isOpened():
        print("無法開啟相機")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = img.shape
            landmarks = results.pose_landmarks.landmark
            pose_dict = {mp_pose.PoseLandmark(i).name: (lm.x, lm.y, lm.z) for i, lm in enumerate(landmarks)}

            analysis = analyze_arm_positions(pose_dict, w, h)

            # 顯示各部位判斷結果
            for i, line in enumerate(analysis):
                if "Other" in line or "Error" in line:
                    color = (0, 0, 0)       # Black
                elif "Forward" in line:
                    color = (255, 255, 255) # White
                elif "Right" in line:
                    color = (0, 255, 0)     # Green
                elif "Left" in line:
                    color = (255, 0, 0)     # Blue
                else:
                    color = (128, 128, 128) # Gray

                cv2.putText(img, line, (10, 90 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            direction, head_color = get_head_direction(pose_dict)
            cv2.putText(img, f"Head: {direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)

            detected_poses = detect_traffic_pose(analysis, direction)

            base_y = h - 30  
            for i, pose_name in enumerate(reversed(detected_poses)):
                cv2.putText(img, f"Detected Pose: {pose_name}", (10, base_y - i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Traffic Gesture Detection', img)
            if cv2.waitKey(5) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
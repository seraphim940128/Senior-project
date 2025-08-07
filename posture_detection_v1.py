import cv2
import numpy as np
import mediapipe as mp
import time
import math

# 數學函式
def get_angle(a, b, c, w, h):
    a = (a[0] * w, a[1] * h, a[2])
    b = (b[0] * w, b[1] * h, b[2])
    c = (c[0] * w, c[1] * h, c[2])

    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def get_point(pose, name, landmarks=None, visibility_threshold=0.5):
    if name in pose:
        point = np.array(pose[name])
        return point
    return None

def get_angle_label(angle, up_range, horizontal_range):
    if up_range[0] <= angle <= up_range[1]:
        return "Raised Up"
    elif horizontal_range[0] <= angle <= horizontal_range[1]:
        return "Extended Horizontally"
    return "Other"

def get_angle_to_horizontal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    return math.degrees(math.atan2(dy, dx))

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

def get_distance(a, b, w, h):
    a = (a[0] * w, a[1] * h)
    b = (b[0] * w, b[1] * h)
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_straight(shoulder, elbow, wrist, tolerance=0.1):
    a = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    b = np.linalg.norm(np.array(elbow) - np.array(wrist))
    c = np.linalg.norm(np.array(shoulder) - np.array(wrist))
    return abs((a + b) - c) < tolerance

def angle_between(p1, p2, axis='x'):
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) if axis == 'x' \
           else np.degrees(np.arctan2(p2[0]-p1[0], p2[1]-p1[1]))

def detect_pose(analysis, head_direction, target_pose_name="Shoulder Forward Elevation"):
    pose_map = {
        "Shoulder Forward Elevation": [
            "Right Forearm: Fully Raised",
            "Left Forearm: Fully Raised",
            "Right Arm: Raised Up",
            "Left Arm: Raised Up",
            "Arms Above Head"
        ],
    }
    
    keywords = pose_map.get(target_pose_name, [])
    
    all_features = analysis + [f"Head: {head_direction}"]
    
    if all(k in all_features for k in keywords):
        return True
    return False

def analyze_arm_positions(pose, w, h):
    results = []

    # Right Arm
    shoulder_r = get_point(pose, "RIGHT_SHOULDER")
    elbow_r    = get_point(pose, "RIGHT_ELBOW")
    wrist_r    = get_point(pose, "RIGHT_WRIST")
    if shoulder_r is not None and elbow_r is not None:
        angle_r = angle_between(shoulder_r, elbow_r)
        results.append(f"Right Arm: {get_angle_label(angle_r, (-135, -45), (-30, 30))}")
    else:
        results.append("Right Arm: Other")

    # Left Arm
    shoulder_l = get_point(pose, "LEFT_SHOULDER")
    elbow_l    = get_point(pose, "LEFT_ELBOW")
    wrist_l    = get_point(pose, "LEFT_WRIST")
    if shoulder_l is not None and elbow_l is not None:
        angle_l = angle_between(shoulder_l, elbow_l)
        results.append(f"Left Arm: {get_angle_label(angle_l, (-135, -45), (-30, 30))}")
    else:
        results.append("Left Arm: Other")

    # Right Upper Arm
    if shoulder_r is not None and elbow_r is not None:
        dx, dy = abs(elbow_r[0]-shoulder_r[0]), abs(elbow_r[1]-shoulder_r[1])
        results.append("Right Upper Arm: Extended Right" if dx > 0.05 and dy < 0.08
                       else "Right Upper Arm: Other")
    else:
        results.append("Right Upper Arm: Other")

    # Left Upper Arm
    if shoulder_l is not None and elbow_l is not None:
        dx, dy = abs(elbow_l[0]-shoulder_l[0]), abs(elbow_l[1]-shoulder_l[1])
        results.append("Left Upper Arm: Extended Left" if dx > 0.05 and dy < 0.08
                       else "Left Upper Arm: Other")
    else:
        results.append("Left Upper Arm: Other")

    # Half‑Raised 判定
    if shoulder_r is not None and elbow_r is not None:
        angle = get_angle_to_horizontal(shoulder_r, elbow_r)
        results.append("Right Arm: Half-Raised" if -135 < angle < -115
                       else "Right Arm: Not Half-Raised")
    else:
        results.append("Right Arm: Half-Raised: Not Visible")

    if shoulder_l is not None and elbow_l is not None:
        angle = get_angle_to_horizontal(shoulder_l, elbow_l)
        results.append("Left Arm: Half-Raised" if -135 < angle < -115
                       else "Left Arm: Not Half-Raised")
    else:
        results.append("Left Arm: Half-Raised: Not Visible")

    # 前臂彎曲角度
    if shoulder_r is not None and elbow_r is not None and wrist_r is not None:
        forearm_angle_r = get_angle(shoulder_r, elbow_r, wrist_r, w, h)
        results.append("Right Forearm: Fully Raised"   if forearm_angle_r > 130
                       else "Right Forearm: Partially Raised" if forearm_angle_r > 60
                       else "Right Forearm: Other")
    else:
        results.append("Right Forearm: Other")

    if shoulder_l is not None and elbow_l is not None and wrist_l is not None:
        forearm_angle_l = get_angle(shoulder_l, elbow_l, wrist_l, w, h)
        results.append("Left Forearm: Fully Raised"   if forearm_angle_l > 130
                       else "Left Forearm: Partially Raised" if forearm_angle_l > 60
                       else "Left Forearm: Other")
    else:
        results.append("Left Forearm: Other")

    # 手掌方向 - Using Z-coordinate difference for depth check
    right_index = get_point(pose, "RIGHT_INDEX")
    if wrist_r is not None and right_index is not None:
        results.append("Right Palm: Facing Forward" if wrist_r[2]-right_index[2] > 0.02
                       else "Right Palm: Other")
    else:
        results.append("Right Palm: Other")

    left_index = get_point(pose, "LEFT_INDEX")
    if wrist_l is not None and left_index is not None:
        results.append("Left Palm: Facing Forward" if wrist_l[2]-left_index[2] > 0.02
                       else "Left Palm: Other")
    else:
        results.append("Left Palm: Other")

    # 手是否高過頭
    nose = get_point(pose, "NOSE")
    if wrist_l is not None and wrist_r is not None and nose is not None:
        results.append("Arms Above Head" if wrist_l[1] < nose[1] and wrist_r[1] < nose[1]
                       else "Arms Not Above Head")
    else:
        results.append("Arms Above Head: Not Visible")

    return results


#共五個動作: elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_forward_elevation, side_tap

def elbow_flexion(shoulder, elbow, wrist, w, h, img, elbow_init=(0,0,0), step=-1, round=0, start=False, start_time=0):
    max_upper_arm_shift, max_upper_elbow_shift = 0.25, 0.08
    if abs(elbow[0] - shoulder[0]) > max_upper_arm_shift or abs(elbow[1] - elbow_init[1]) > max_upper_elbow_shift:
        #cv2.putText(img, "Keep upper arm still", (int(w/3), h-30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time, elbow_init, "請固定上臂"
    
    angle_elbow = get_angle(shoulder, elbow, wrist, w, h)
    current_time = time.time()

    if step == -1 and 160 < angle_elbow < 185:
        step, start_time, start = 0, current_time, True
        elbow_init = elbow
        return step, round, start, start_time, elbow_init, "開始動作 (手臂伸直)"
    
    elif step == 0 and 40 < angle_elbow < 80 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
        return step, round, start, start_time, elbow_init, "請持續彎曲手肘"
    
    elif step == 1 and 160 < angle_elbow < 185 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            return step, round, start, start_time, elbow_init, f"完成第 {round} 次！"
        return step, round, start, start_time, elbow_init, "請完全伸直手臂"
    
    return step, round, start, start_time, elbow_init, ""

def shoulder_flexion(landmarks, side, w, h, step, round, start, start_time):
    current_time = time.time()
    
    if side == "left":
        shoulder = landmarks["LEFT_SHOULDER"]
        elbow = landmarks["LEFT_ELBOW"]
        wrist = landmarks["LEFT_WRIST"]
        hip = landmarks["LEFT_HIP"]
    else:
        shoulder = landmarks["RIGHT_SHOULDER"]
        elbow = landmarks["RIGHT_ELBOW"]
        wrist = landmarks["RIGHT_WRIST"]
        hip = landmarks["RIGHT_HIP"]
    
    if not is_straight(shoulder, elbow, wrist):
        return step, round, start, start_time, "Please stretch your arms"
    
    max_horizontal_diff = 0.2
    horizontal_diff = abs(wrist[0] - shoulder[0])
    if horizontal_diff > max_horizontal_diff:
        return step, round, start, start_time, "Not a shoulder flexion(請直舉而非側舉)"

    angle_shoulder = get_angle(wrist, shoulder, hip, w, h)

    if step == -1 and 10 < angle_shoulder < 30:
        step, start_time, start = 0, current_time, True

    elif step == 0 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time

    elif step == 1 and 160 < angle_shoulder < 190 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time

    elif step == 2 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time

    elif step == 3 and 10 < angle_shoulder < 30 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

def shoulder_abduction(landmarks, side, w, h, step, round, start, start_time):
    current_time = time.time()
    
    if side == "left":
        shoulder = landmarks["LEFT_SHOULDER"]
        elbow = landmarks["LEFT_ELBOW"]
        wrist = landmarks["LEFT_WRIST"]
        hip = landmarks["LEFT_HIP"]
    else:
        shoulder = landmarks["RIGHT_SHOULDER"]
        elbow = landmarks["RIGHT_ELBOW"]
        wrist = landmarks["RIGHT_WRIST"]
        hip = landmarks["RIGHT_HIP"]

    angle_shoulder = get_angle(hip, shoulder, elbow, w, h)
    angle_elbow = get_angle(shoulder, elbow, wrist, w, h)

    if not is_straight(shoulder, elbow, wrist):
        return step, round, start, start_time, "Please stretch your arms"
    
    max_depth_diff = 0.2
    depth_diff = abs((wrist[2]-hip[2]) - (shoulder[2]-hip[2]))
    if depth_diff > max_depth_diff:
        return step, round, start, start_time, "Not a shoulder abduction (請側舉而非直舉)"

    if step == -1 and 10 < angle_shoulder < 30:
        step, start_time, start = 0, current_time, True

    elif step == 0 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time

    elif step == 1 and 160 < angle_shoulder < 190 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time

    elif step == 2 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time

    elif step == 3 and 10 < angle_shoulder < 30 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

HOLD_SEC    = 1.0   
RELEASE_SEC = 1.0 
def shoulder_forward_elevation_logic(landmarks, w, h, step, round, start, start_time):
    current_time = time.time()

    shoulder_l = landmarks.get("LEFT_SHOULDER")
    elbow_l = landmarks.get("LEFT_ELBOW")
    wrist_l = landmarks.get("LEFT_WRIST")
    hip_l = landmarks.get("LEFT_HIP")

    shoulder_r = landmarks.get("RIGHT_SHOULDER")
    elbow_r = landmarks.get("RIGHT_ELBOW")
    wrist_r = landmarks.get("RIGHT_WRIST")
    hip_r = landmarks.get("RIGHT_HIP")

    if None in [shoulder_l, elbow_l, wrist_l, hip_l, shoulder_r, elbow_r, wrist_r, hip_r]:
        return step, round, start, start_time, "偵測不到完整左右手臂"

    if not is_straight(shoulder_l, elbow_l, wrist_l) or not is_straight(shoulder_r, elbow_r, wrist_r):
        return step, round, start, start_time, "請伸直雙手"

    max_side_diff = 0.2
    if abs(wrist_l[0] - shoulder_l[0]) > max_side_diff or abs(wrist_r[0] - shoulder_r[0]) > max_side_diff:
        return step, round, start, start_time, "請雙手直舉前抬，勿側抬"

    angle_l = get_angle(wrist_l, shoulder_l, hip_l, w, h)
    angle_r = get_angle(wrist_r, shoulder_r, hip_r, w, h)

    if step == -1 and 10 < angle_l < 30 and 10 < angle_r < 30:
        step, start_time, start = 0, current_time, True
    elif step == 0 and 80 < angle_l < 100 and 80 < angle_r < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
    elif step == 1 and 160 < angle_l < 190 and 160 < angle_r < 190 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time
    elif step == 2 and 80 < angle_l < 100 and 80 < angle_r < 100 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time
    elif step == 3 and 10 < angle_l < 30 and 10 < angle_r < 30 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            return step, round, start, start_time, f"完成第 {round} 次！"

    return step, round, start, start_time, ""

def side_tap(landmarks, side, w, h, step, round, start, start_time):
    STRENGTH_LOWERBOUND = 1.5
    STRENGTH_INTENSITY = 2.0

    current_time = time.time()
    
    left_hip = landmarks["LEFT_HIP"]
    left_knee = landmarks["LEFT_KNEE"]
    left_ankle = landmarks["LEFT_ANKLE"]
    right_hip = landmarks["RIGHT_HIP"]
    right_knee = landmarks["RIGHT_KNEE"]
    right_ankle = landmarks["RIGHT_ANKLE"]

    hip_width = get_distance(left_hip, right_hip, w, h)
    left_foot_to_right_foot = get_distance(left_ankle, right_ankle, w, h)
    
    if not is_straight(left_hip, left_knee, left_ankle) or not is_straight(right_hip, right_knee, right_ankle):
        return step, round, start, start_time, "Keep legs straight"
    
    if side == "left":
        foot_to_center_x = abs(left_ankle[0] - (left_hip[0] + right_hip[0])/2) * w
    else:
        foot_to_center_x = abs(right_ankle[0] - (left_hip[0] + right_hip[0])/2) * w

    if step == -1 and left_foot_to_right_foot < hip_width * STRENGTH_LOWERBOUND:
        step, start_time, start = 0, current_time, True
    
    elif step == 0 and foot_to_center_x > hip_width * STRENGTH_INTENSITY and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
    
    elif step == 1 and left_foot_to_right_foot < hip_width * STRENGTH_LOWERBOUND and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

import string

# 所有動作清單
ALL_EXERCISES = [
    "side_tap_left",
    "side_tap_right",
    "shoulder_abduction_left",
    "shoulder_abduction_right",
    "shoulder_flexion_left",
    "shoulder_flexion_right",
    "shoulder_forward_elevation",
    "elbow_flexion_left",
    "elbow_flexion_right"
]

# 動作對應顏色
EXERCISE_COLORS = {
    "side_tap_left": (255, 0, 0),
    "side_tap_right": (255, 0, 0),
    "shoulder_abduction_left": (0, 255, 0),
    "shoulder_abduction_right": (0, 255, 0),
    "shoulder_flexion_left": (0, 0, 255),
    "shoulder_flexion_right": (0, 0, 255),
    "shoulder_forward_elevation": (255, 64, 128),
    "elbow_flexion_left": (128, 0, 128),
    "elbow_flexion_right": (128, 0, 128),
}

def select_exercises():
    print("\n請選擇要偵測的動作")
    for idx, name in enumerate(ALL_EXERCISES):
        print(f"{string.ascii_uppercase[idx]}. {name}")
    choice = input("輸入編號：").upper().strip()

    selected = []
    for ch in choice:
        if ch in string.ascii_uppercase[:len(ALL_EXERCISES)]:
            idx = string.ascii_uppercase.index(ch)
            selected.append(ALL_EXERCISES[idx])

    if not selected:
        selected = ALL_EXERCISES[:]

    print(f"目前選擇動作: {', '.join(selected)}")
    return selected

def init_exercises(selected):
    exercises = {}
    for name in selected:
        exercises[name] = {
            "step": -1,
            "round": 0,
            "start": False,
            "start_time": 0,
            "color": EXERCISE_COLORS.get(name, (255, 255, 255))
        }
        if "elbow_flexion" in name:
            exercises[name]["elbow_init"] = (0, 0, 0)
    return exercises

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟相機")
        return

    # 初次選擇動作
    selected_exercises = select_exercises()
    exercises = init_exercises(selected_exercises)

    with mp_pose.Pose(min_detection_confidence=0.45, min_tracking_confidence=0.45) as pose:
        while True:
            ret, img = cap.read()
            if not ret:
                print("無法讀取影像")
                break
            img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img2)
            h, w, _ = img.shape

            current_frame_error_message = ""

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                landmarks = {
                    mp_pose.PoseLandmark(i).name: (lm.x, lm.y, lm.z)
                    for i, lm in enumerate(results.pose_landmarks.landmark)
                }

                for name in list(exercises.keys()):
                    params = {key: exercises[name][key] for key in ['step', 'round', 'start', 'start_time']}

                    if "elbow_flexion" in name:
                        side = "left" if "left" in name else "right"
                        shoulder = landmarks[f"{side.upper()}_SHOULDER"]
                        elbow    = landmarks[f"{side.upper()}_ELBOW"]
                        wrist    = landmarks[f"{side.upper()}_WRIST"]
                        elbow_init = elbow if exercises[name]["elbow_init"] == (0, 0, 0) else exercises[name]["elbow_init"]
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], exercises[name]["elbow_init"], current_frame_error_message = \
                        elbow_flexion(shoulder, elbow, wrist, w, h, img, elbow_init, **params)
                    
                    elif "side_tap" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            side_tap(landmarks, side, w, h, **params)
                    
                    elif "shoulder_abduction" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_abduction(landmarks, side, w, h, **params)
                    
                    elif "shoulder_flexion" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_flexion(landmarks, side, w, h, **params)

                    elif "shoulder_forward_elevation" in name:
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_forward_elevation_logic(landmarks, w, h, **params)

            y_offset = 30
            for name, data in exercises.items():
                color = data["color"]
                cv2.putText(img, f'{name.replace("_", " ").title()}: Step {data["step"]}, Round {data["round"]}', 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30

            cv2.namedWindow('Posture Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Posture Detection', 960, 720)
            cv2.imshow('Posture Detection', img)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # 重新選擇動作
                selected_exercises = select_exercises()
                exercises = init_exercises(selected_exercises)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
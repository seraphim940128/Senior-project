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


#共五個動作: elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_forward_elevation, side_tap

def elbow_flexion(landmarks, side, w, h, current_time, step, round, start, start_time):
    if side == "left":
        shoulder = landmarks["LEFT_SHOULDER"]
        elbow = landmarks["LEFT_ELBOW"]
        wrist = landmarks["LEFT_WRIST"]
    else:
        shoulder = landmarks["RIGHT_SHOULDER"]
        elbow = landmarks["RIGHT_ELBOW"]
        wrist = landmarks["RIGHT_WRIST"]

    # 判斷上臂是否固定（避免大幅移動）
    max_upper_arm_shift = 0.25
    if abs(elbow[0] - shoulder[0]) > max_upper_arm_shift:
        return step, round, start, start_time, "Please keep your upper arm still"

    angle = get_angle(shoulder, elbow, wrist, w, h)

    if step == -1:
        step, start_time, start = 0, current_time, True
        return step, round, start, start_time, "Starting movement (arm extended)"

    elif step == 0 and 40 < angle < 80 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
        return step, round, start, start_time, "Keep bending your elbow"

    elif step == 1 and 160 < angle < 185 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
            return step, round, start, start_time, f"Completed round {round}!"
        return step, round, start, start_time, "Please fully extend your arm"

    return step, round, start, start_time, ""


def shoulder_flexion(landmarks, side, w, h, current_time, step, round, start, start_time):
    
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
        return step, round, start, start_time, "Not a shoulder flexion"

    angle_shoulder = get_angle(wrist, shoulder, hip, w, h)

    if step == -1:
        step, start_time, start = 0, current_time, True

    elif step == 0 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time

    elif step == 1 and 160 < angle_shoulder < 190 and start:
        if current_time - start_time > 1:
            step, start_time = 2, current_time

    elif step == 2 and 75 < angle_shoulder < 105 and start:
        if current_time - start_time > 1:
            step, start_time = 3, current_time

    elif step == 3 and 0 < angle_shoulder < 40 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

def shoulder_abduction(landmarks, side, w, h, current_time, step, round, start, start_time):

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

    if not is_straight(shoulder, elbow, wrist):
        return step, round, start, start_time, "Please stretch your arms"
    
    max_depth_diff = 0.2
    depth_diff = abs((wrist[2]-hip[2]) - (shoulder[2]-hip[2]))
    if depth_diff > max_depth_diff:
        return step, round, start, start_time, "Not a shoulder abduction"

    if step == -1:
        step, start_time, start = 0, current_time, True

    elif step == 0 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time

    elif step == 1 and 160 < angle_shoulder < 190 and start:
        if current_time - start_time > 1:
            step, start_time = 2, current_time

    elif step == 2 and 80 < angle_shoulder < 100 and start:
        if current_time - start_time > 1:
            step, start_time = 3, current_time

    elif step == 3 and 10 < angle_shoulder < 30 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

def shoulder_forward_elevation(landmarks, w, h, current_time, step, round, start, start_time):


    shoulder_l = landmarks.get("LEFT_SHOULDER")
    elbow_l = landmarks.get("LEFT_ELBOW")
    wrist_l = landmarks.get("LEFT_WRIST")
    hip_l = landmarks.get("LEFT_HIP")

    shoulder_r = landmarks.get("RIGHT_SHOULDER")
    elbow_r = landmarks.get("RIGHT_ELBOW")
    wrist_r = landmarks.get("RIGHT_WRIST")
    hip_r = landmarks.get("RIGHT_HIP")

    if None in [shoulder_l, elbow_l, wrist_l, hip_l, shoulder_r, elbow_r, wrist_r, hip_r]:
        return step, round, start, start_time, "Cannot detect full left and right arms"

    if not is_straight(shoulder_l, elbow_l, wrist_l) or not is_straight(shoulder_r, elbow_r, wrist_r):
        return step, round, start, start_time, "Please keep your arms straight"

    max_side_diff = 0.3
    if abs(wrist_l[0] - shoulder_l[0]) > max_side_diff or abs(wrist_r[0] - shoulder_r[0]) > max_side_diff:
        return step, round, start, start_time, "Please raise both hands forward, not to the side"

    angle_l = get_angle(wrist_l, shoulder_l, hip_l, w, h)
    angle_r = get_angle(wrist_r, shoulder_r, hip_r, w, h)

    if step == -1:
        step, start_time, start = 0, current_time, True
    elif step == 0 and 75 < angle_l < 100 and 75 < angle_r < 100 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
    elif step == 1 and 160 < angle_l < 190 and 160 < angle_r < 190 and start:
        if current_time - start_time > 1:
            step, start_time = 2, current_time
    elif step == 2 and 75 < angle_l < 100 and 75 < angle_r < 100 and start:
        if current_time - start_time > 1:
            step, start_time = 3, current_time
    elif step == 3 and 0 < angle_l < 45 and 0 < angle_r < 45 and start:
        if current_time - start_time > 1:
            round += 1
            step, start = -1, False
            return step, round, start, start_time, f"Completed round {round}!"

    return step, round, start, start_time, ""

def side_tap(landmarks, side, w, h, current_time, step, round, start, start_time):
    STRENGTH_LOWERBOUND = 1.5
    STRENGTH_INTENSITY = 2.0
    
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
    return exercises

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture("elbow_flexion_left_50.mp4")

    if not cap.isOpened():
        print("無法開啟相機")
        return

    # 取得影片 FPS 與每幀時間
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # 預設值
    frame_time = 1.0 / fps
    video_time = 0.0

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

            # 更新影片時間
            video_time += frame_time

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
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            elbow_flexion(landmarks, side, w, h, video_time, **params)
                    
                    elif "side_tap" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            side_tap(landmarks, side, w, h, video_time, **params)
                    
                    elif "shoulder_abduction" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_abduction(landmarks, side, w, h, video_time, **params)
                    
                    elif "shoulder_flexion" in name:
                        side = "left" if "left" in name else "right"
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_flexion(landmarks, side, w, h, video_time, **params)

                    elif "shoulder_forward_elevation" in name:
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            shoulder_forward_elevation(landmarks, w, h, video_time, **params)

            # 畫出當前動作資訊
            y_offset = 30
            for name, data in exercises.items():
                color = data["color"]
                cv2.putText(img, f'{name.replace("_", " ").title()}: Step {data["step"]}, Round {data["round"]}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30
                if current_frame_error_message:
                    cv2.putText(img, current_frame_error_message, (10, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 顯示影像並保持比例
            h, w, _ = img.shape
            display_w = 960
            scale = display_w / w
            display_h = int(h * scale)
            resized_img = cv2.resize(img, (display_w, display_h))

            cv2.imshow('Posture Detection', resized_img)

            key = cv2.waitKey(3) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                selected_exercises = select_exercises()
                exercises = init_exercises(selected_exercises)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

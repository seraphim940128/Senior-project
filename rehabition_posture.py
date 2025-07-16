import cv2
import numpy as np
import mediapipe as mp
import time
import math

#共五個動作: shoulder_abduction, shoulder_flexion, side_tap, elbow_flexion, shoulder_forward_elevation

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

def get_distance(a, b, w, h):
    a = (a[0] * w, a[1] * h)
    b = (b[0] * w, b[1] * h)
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_arm_straight(shoulder, elbow, wrist, tolerance=0.1):
    a = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    b = np.linalg.norm(np.array(elbow) - np.array(wrist))
    c = np.linalg.norm(np.array(shoulder) - np.array(wrist))
    return abs((a + b) - c) < tolerance

def side_tap(landmarks, side, w, h, step, round, start, start_time):
    current_time = time.time()
    
    left_hip = landmarks["LEFT_HIP"]
    left_knee = landmarks["LEFT_KNEE"]
    left_ankle = landmarks["LEFT_ANKLE"]
    right_hip = landmarks["RIGHT_HIP"]
    right_knee = landmarks["RIGHT_KNEE"]
    right_ankle = landmarks["RIGHT_ANKLE"]

    hip_width = get_distance(left_hip, right_hip, w, h)
    left_foot_to_right_foot = get_distance(left_ankle, right_ankle, w, h)
    
    left_knee_angle = get_angle(left_hip, left_knee, left_ankle, w, h)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle, w, h)
    
    if not (150 < left_knee_angle < 210 and 150 < right_knee_angle < 210):
        return step, round, start, start_time, "Keep legs straight"
    
    if side == "left":
        foot_to_center = abs(left_ankle[0] - (left_hip[0] + right_hip[0])/2) * w
    else:
        foot_to_center = abs(right_ankle[0] - (left_hip[0] + right_hip[0])/2) * w

    if step == -1 and left_foot_to_right_foot < hip_width * 1.5:
        step, start_time, start = 0, current_time, True
    
    elif step == 0 and foot_to_center > hip_width * 2.4 and start:
        if current_time - start_time > 1:
            step, start_time = 1, current_time
    
    elif step == 1 and left_foot_to_right_foot < hip_width * 1.5 and start:
        if current_time - start_time > 1:
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

    if not (160 < angle_elbow < 200):
        return step, round, start, start_time, "Please stretch your arms"
    
    if step == -1 and 15 < angle_shoulder < 25:
        step, start_time, start = 0, current_time, True

    elif step == 0 and 85 < angle_shoulder < 95 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time

    elif step == 1 and 170 < angle_shoulder < 190 and start:
        if current_time - start_time > 2:
            step, start_time = 2, current_time

    elif step == 2 and 85 < angle_shoulder < 95 and start:
        if current_time - start_time > 2:
            step, start_time = 3, current_time

    elif step == 3 and 5 < angle_shoulder < 15 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
    
    return step, round, start, start_time, ""

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
    
    if not is_arm_straight(shoulder, elbow, wrist):
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


def calculate_angle_sfe(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

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

def get_point(pose, name, landmarks=None, visibility_threshold=0.5):
    if name in pose:
        point = np.array(pose[name])
        
        return point
    return None

def angle_between(p1, p2, axis='x'):
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) if axis == 'x' \
           else np.degrees(np.arctan2(p2[0]-p1[0], p2[1]-p1[1]))

def get_angle_label(angle, up_range, horizontal_range):
    if up_range[0] <= angle <= up_range[1]:
        return "Raised Up"
    elif horizontal_range[0] <= angle <= horizontal_range[1]:
        return "Extended Horizontally"
    return "Other"

def elbow_flexion(shoulder, elbow, wrist, w, h, img, step=-1, round=0, start=False, start_time=0):
    max_upper_arm_shift = 0.15
    if abs(elbow[0] - shoulder[0]) > max_upper_arm_shift:
        #cv2.putText(img, "Keep upper arm still", (int(w/3), h-30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return step, round, start, start_time, "請固定上臂"
    
    angle_elbow = get_angle(shoulder, elbow, wrist, w, h)
    current_time = time.time()

    if step == -1 and 160 < angle_elbow < 185:
        step, start_time, start = 0, current_time, True
        return step, round, start, start_time, "開始動作 (手臂伸直)"
    
    elif step == 0 and 40 < angle_elbow < 80 and start:
        if current_time - start_time > 2:
            step, start_time = 1, current_time
        return step, round, start, start_time, "請持續彎曲手肘"
    
    elif step == 1 and 160 < angle_elbow < 185 and start:
        if current_time - start_time > 2:
            round += 1
            step, start = -1, False
            return step, round, start, start_time, f"完成第 {round} 次！"
        return step, round, start, start_time, "請完全伸直手臂"
    
    return step, round, start, start_time, ""


def analyze_arm_positions(pose, mp_pose_landmarks):
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
        forearm_angle_r = calculate_angle_sfe(shoulder_r, elbow_r, wrist_r)
        results.append("Right Forearm: Fully Raised"   if forearm_angle_r > 130
                       else "Right Forearm: Partially Raised" if forearm_angle_r > 60
                       else "Right Forearm: Other")
    else:
        results.append("Right Forearm: Other")

    if shoulder_l is not None and elbow_l is not None and wrist_l is not None:
        forearm_angle_l = calculate_angle_sfe(shoulder_l, elbow_l, wrist_l)
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

HOLD_SEC    = 1.0   
RELEASE_SEC = 1.0 
def shoulder_forward_elevation_logic(landmarks, w, h,
                                     step, round, start, start_time):
    now = time.time()
    analysis  = analyze_arm_positions(landmarks, None)
    head_dir, _ = get_head_direction(landmarks)
    pose_up  = detect_pose(analysis, head_dir,
                           target_pose_name="Shoulder Forward Elevation")
    error_msg = ""

    if step == -1:
        if pose_up:                       
            step       = 0
            start_time = now             

    
    elif step == 0:
        if pose_up:
            if now - start_time >= HOLD_SEC:
                
                error_msg = "可以放下手臂"
        else:  
            step       = 1
            start_time = now             
            error_msg  = "偵測到放下，請維持"
    
    elif step == 1:
        if not pose_up:
            if now - start_time >= RELEASE_SEC:
                round += 1               
                step  = -1
                error_msg = f"第 {round} 回合完成！"
        else:  
            step       = 0
            start_time = now

    return step, round, start, start_time, error_msg


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟相機")
        return

    
    exercises = {
        "side_tap_left": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (255, 0, 0)},
        "side_tap_right": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (255, 0, 0)},
        "shoulder_abduction_left": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (0, 255, 0)},
        "shoulder_abduction_right": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (0, 255, 0)},
        "shoulder_flexion_left": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (0, 0, 255)},
        "shoulder_flexion_right": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (0, 0, 255)},
        "shoulder_forward_elevation": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (255, 255, 0)},
        "elbow_flexion_left": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (128, 0, 128)},
        "elbow_flexion_right": {"step": -1, "round": 0, "start": False, "start_time": 0, "color": (128, 0, 128)},
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, img = cap.read()
            if not ret:
                print("無法讀取影像")
                break

            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img2)
            h, w, _ = img.shape
            
            current_frame_error_message = "" 

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                landmarks = {
                    mp_pose.PoseLandmark(i).name: (lm.x, lm.y, lm.z)
                    for i, lm in enumerate(results.pose_landmarks.landmark)
                }

                
                for name in exercises:
                    params = {key: exercises[name][key] for key in ['step', 'round', 'start', 'start_time']}
                    
                    if "side_tap" in name:
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
                    elif "elbow_flexion" in name:
                        side = "left" if "left" in name else "right"
                        shoulder = landmarks[f"{side.upper()}_SHOULDER"]
                        elbow    = landmarks[f"{side.upper()}_ELBOW"]
                        wrist    = landmarks[f"{side.upper()}_WRIST"]
                        exercises[name]["step"], exercises[name]["round"], \
                        exercises[name]["start"], exercises[name]["start_time"], current_frame_error_message = \
                            elbow_flexion(shoulder, elbow, wrist, w, h, img, **params)
            
            y_offset = 30
            for name, data in exercises.items():
                color = data["color"]
                cv2.putText(img, f'{name.replace("_", " ").title()}: Step {data["step"]}, Round {data["round"]}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30

            #if current_frame_error_message:
                # cv2.putText(img, current_frame_error_message, (int(w/3), h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.namedWindow('Posture Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Posture Detection', img)

            if cv2.waitKey(5) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
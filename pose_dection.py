import cv2
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

        img = cv2.resize(img, (520, 300))
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

            #X:[0] Y:[1] Z:[2] 
            #底下是所有關鍵點的座標
            """
            print("鼻子座標：", NOSE)
            print("左眼內側座標：", LEFT_EYE_INNER)
            print("左眼座標：", LEFT_EYE)
            print("左眼外側座標：", LEFT_EYE_OUTER)
            print("右眼內側座標：", RIGHT_EYE_INNER)
            print("右眼座標：", RIGHT_EYE)
            print("右眼外側座標：", RIGHT_EYE_OUTER)
            print("左耳座標：", LEFT_EAR)
            print("右耳座標：", RIGHT_EAR)
            print("嘴巴左側座標：", MOUTH_LEFT)
            print("嘴巴右側座標：", MOUTH_RIGHT)
            print("左肩座標：", LEFT_SHOULDER)
            print("右肩座標：", RIGHT_SHOULDER)
            print("左手肘座標：", LEFT_ELBOW)
            print("右手肘座標：", RIGHT_ELBOW)
            print("左手腕座標：", LEFT_WRIST)
            print("右手腕座標：", RIGHT_WRIST)
            print("左手拇指座標：", LEFT_THUMB)
            print("右手拇指座標：", RIGHT_THUMB)
            print("左手食指座標：", LEFT_INDEX)
            print("右手食指座標：", RIGHT_INDEX)
            print("左手小指座標：", LEFT_PINKY)
            print("右手小指座標：", RIGHT_PINKY)
            print("左臀座標：", LEFT_HIP)
            print("右臀座標：", RIGHT_HIP)
            print("左膝蓋座標：", LEFT_KNEE)
            print("右膝蓋座標：", RIGHT_KNEE)
            print("左腳踝座標：", LEFT_ANKLE)
            print("右腳踝座標：", RIGHT_ANKLE)
            print("左腳跟座標：", LEFT_HEEL)
            print("右腳跟座標：", RIGHT_HEEL)
            print("左腳尖座標：", LEFT_FOOT_INDEX)
            print("右腳尖座標：", RIGHT_FOOT_INDEX)
            """



            # 偵測左右手是否舉起來
            def is_arm_raised(shoulder, wrist, threshold=0.02):
                return wrist[1] < shoulder[1] - threshold

            left_raised = is_arm_raised(LEFT_SHOULDER, LEFT_WRIST)
            right_raised = is_arm_raised(RIGHT_SHOULDER, RIGHT_WRIST)

            if left_raised:
                cv2.putText(img, 'Left arm is raised', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if right_raised:
                cv2.putText(img, 'Right arm is raised', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if left_raised and right_raised:
                cv2.putText(img, 'Both arms are raised!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

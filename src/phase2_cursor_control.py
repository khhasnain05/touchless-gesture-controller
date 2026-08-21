import cv2
import time
import pyautogui
import numpy as np
import mediapipe as mp

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

margin = 100 
prev_x, prev_y = 0, 0
smooth_factor = 5
prev_time = 0

print("Phase 2 Fixed: Dynamic Resolution Active.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    
    # DYNAMIC RESOLUTION FIX: Get true frame height & width
    cam_h, cam_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]

            raw_cam_x = int(index_tip.x * cam_w)
            raw_cam_y = int(index_tip.y * cam_h)

            cv2.circle(frame, (raw_cam_x, raw_cam_y), 10, (0, 255, 255), cv2.FILLED)

            # Accurate bounding box mapping
            target_x = np.interp(raw_cam_x, [margin, cam_w - margin], [0, screen_w])
            target_y = np.interp(raw_cam_y, [margin, cam_h - margin], [0, screen_h])

            if prev_x == 0 and prev_y == 0:
                curr_x, curr_y = target_x, target_y
            else:
                curr_x = prev_x + (target_x - prev_x) / smooth_factor
                curr_y = prev_y + (target_y - prev_y) / smooth_factor

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

    # Rectangle now scales correctly to any webcam resolution
    cv2.rectangle(frame, (margin, margin), (cam_w - margin, cam_h - margin), (255, 0, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Mode: Cursor Motion", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("AI Air Mouse - Phase 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
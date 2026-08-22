import cv2
import time
import math
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
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

cap = cv2.VideoCapture(0)

# Tuned parameters for maximum stability
MARGIN_X = 140           # Active tracking box margin X
MARGIN_Y = 120           # Active tracking box margin Y
SMOOTHING = 8            # Higher = smoother & easier to target small buttons
PINCH_THRESH = 30        # Pinch distance threshold
CLICK_COOLDOWN = 0.45    # Cooldown between clicks

prev_x, prev_y = 0, 0
last_click_time = 0
is_dragging = False
prev_time = 0

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

print("Phase 3 Precision Mode Active.")
print("Point with INDEX FINGER ONLY to move cursor. Open palm = Rest Mode.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    cam_h, cam_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_mode = "REST MODE (Palm Open / Hand Resting)"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            
            thumb_tip = (int(landmarks[4].x * cam_w), int(landmarks[4].y * cam_h))
            index_tip = (int(landmarks[8].x * cam_w), int(landmarks[8].y * cam_h))
            middle_tip = (int(landmarks[12].x * cam_w), int(landmarks[12].y * cam_h))

            # Finger states (Extended vs Folded)
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_down = landmarks[16].y > landmarks[14].y
            pinky_down = landmarks[20].y > landmarks[18].y

            index_thumb_dist = calculate_distance(index_tip, thumb_tip)
            middle_thumb_dist = calculate_distance(middle_tip, thumb_tip)

            curr_time = time.time()

            # ---------------- GESTURE 1: LEFT CLICK / DRAG ----------------
            if index_thumb_dist < PINCH_THRESH:
                cv2.line(frame, index_tip, thumb_tip, (0, 255, 0), 4)
                
                # FREEZE CURSOR POSITION DURING PINCH to prevent target slipping
                if not is_dragging and (curr_time - last_click_time > CLICK_COOLDOWN):
                    pyautogui.click()
                    last_click_time = curr_time
                    current_mode = "LEFT CLICK"
                else:
                    pyautogui.mouseDown()
                    is_dragging = True
                    current_mode = "DRAGGING..."

            # Release drag when pinch opens
            elif is_dragging:
                pyautogui.mouseUp()
                is_dragging = False

            # ---------------- GESTURE 2: RIGHT CLICK ----------------
            elif middle_thumb_dist < PINCH_THRESH:
                cv2.line(frame, middle_tip, thumb_tip, (0, 0, 255), 4)
                if curr_time - last_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_time = curr_time
                    current_mode = "RIGHT CLICK"

            # ---------------- GESTURE 3: SCROLL MODE ----------------
            # Index + Middle extended UP, Ring + Pinky folded DOWN
            elif index_up and middle_up and ring_down and pinky_down:
                if index_tip[1] < cam_h // 2 - 40:
                    pyautogui.scroll(25)
                    current_mode = "SCROLL UP"
                elif index_tip[1] > cam_h // 2 + 40:
                    pyautogui.scroll(-25)
                    current_mode = "SCROLL DOWN"
                else:
                    current_mode = "SCROLL READY"

            # ---------------- POINTING POSE: CURSOR MOTION ----------------
            # Cursor ONLY moves when Index is UP and Middle finger is DOWN (👉 Pointing Pose)
            elif index_up and not middle_up:
                current_mode = "CURSOR MOVING"
                cv2.circle(frame, index_tip, 10, (0, 255, 255), cv2.FILLED)

                target_x = np.interp(index_tip[0], [MARGIN_X, cam_w - MARGIN_X], [0, screen_w])
                target_y = np.interp(index_tip[1], [MARGIN_Y, cam_h - MARGIN_Y], [0, screen_h])

                if prev_x == 0 and prev_y == 0:
                    curr_x, curr_y = target_x, target_y
                else:
                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

    # Visual Workspace Box
    cv2.rectangle(frame, (MARGIN_X, MARGIN_Y), (cam_w - MARGIN_X, cam_h - MARGIN_Y), (255, 0, 0), 2)

    # Status Bar
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Mode: {current_mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("AI Air Mouse - Phase 3 Precision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Phase 3 complete.")
import os
import cv2
import time
import torch
import torch.nn as nn
import pyautogui
import numpy as np
import mediapipe as mp
from collections import deque

# PyAutoGUI safety configuration
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# -------------------------------------------------------------
# 1. PYTORCH DEEP LEARNING MODEL ARCHITECTURE
# -------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, input_dim=30 * 66, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

MODEL_PATH = os.path.join("..", "models", "best_gesture_model.pth")
device = torch.device("cpu")

model = MLPModel()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded PyTorch Neural Network from: {os.path.abspath(MODEL_PATH)}")
else:
    print(f"❌ ERROR: Model file not found at {os.path.abspath(MODEL_PATH)}")
    exit()

GESTURE_LABELS = {
    0: "idle",
    1: "point",
    2: "left_click",
    3: "right_click",
    4: "scroll_up",
    5: "scroll_down",
}

def normalize_landmarks_hybrid(sequence_np):
    hybrid_sequence = []
    for frame in sequence_np:
        landmarks = frame.reshape(21, 3)
        wrist_abs = landmarks[0].copy()
        
        centered = landmarks - wrist_abs
        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)
        
        shape_norm = centered / max_dist if max_dist > 0 else centered
        combined = np.hstack((shape_norm.flatten(), wrist_abs))
        hybrid_sequence.append(combined)
        
    return np.array(hybrid_sequence, dtype=np.float32)

# -------------------------------------------------------------
# 2. DISCRETE COMMAND CONTROLLER (ZERO CURSOR MOVING)
# -------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=30)

# Debounce Timers (Prevents duplicate key triggers)
ACTION_COOLDOWN = 1.0   # 1 second delay between discrete commands
VOLUME_COOLDOWN = 0.2   # Fast delay for volume stepping

last_action_time = 0
last_volume_time = 0
last_action_text = "SYSTEM STANDBY - READY"

prev_time = 0

print("\n" + "="*60)
print(" 🚀 TOUCHLESS AI GESTURE OS CONTROLLER (PATH 1)")
print("="*60)
print(" • IDLE (Open Palm ✋)  : STANDBY / REST MODE (Zero Action)")
print(" • POINT (👉)         : NEXT SLIDE / SKIP 5s (Right Arrow)")
print(" • RIGHT CLICK (✌️)   : PREV SLIDE / REWIND 5s (Left Arrow)")
print(" • LEFT CLICK (👌)    : PLAY / PAUSE (Spacebar)")
print(" • SCROLL UP          : VOLUME UP (+5%)")
print(" • SCROLL DOWN        : VOLUME DOWN (-5%)")
print("="*60 + "\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    cam_h, cam_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_gesture = "STANDBY"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_frame = []
            for lm in hand_landmarks.landmark:
                raw_frame.extend([lm.x, lm.y, lm.z])
            
            frame_buffer.append(raw_frame)

            # ---------------- DEEP LEARNING INFERENCE ----------------
            if len(frame_buffer) == 30:
                buf_np = np.array(frame_buffer, dtype=np.float32)
                norm_seq = normalize_landmarks_hybrid(buf_np)
                tensor_input = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(tensor_input)
                    probabilities = torch.softmax(outputs, dim=1)
                    conf, pred_class = torch.max(probabilities, 1)
                    
                    current_gesture = GESTURE_LABELS[pred_class.item()]
                    confidence = conf.item() * 100

                curr_time = time.time()

                # ---------------- DISCRETE COMMAND DISPATCHER ----------------
                
                # 1. IDLE (✋) -> REST / NO ACTION
                if current_gesture == "idle":
                    pass

                # 2. POINT (👉) -> NEXT SLIDE / RIGHT ARROW
                elif current_gesture == "point" and confidence > 75:
                    if curr_time - last_action_time > ACTION_COOLDOWN:
                        pyautogui.press('right')
                        last_action_text = "EXECUTE: NEXT (Right Arrow ->)"
                        last_action_time = curr_time
                        print("▶️ Executed: NEXT (Right Arrow)")

                # 3. RIGHT CLICK (✌️) -> PREVIOUS SLIDE / LEFT ARROW
                elif current_gesture == "right_click" and confidence > 75:
                    if curr_time - last_action_time > ACTION_COOLDOWN:
                        pyautogui.press('left')
                        last_action_text = "EXECUTE: PREVIOUS (Left Arrow <-)"
                        last_action_time = curr_time
                        print("◀️ Executed: PREVIOUS (Left Arrow)")

                # 4. LEFT CLICK (👌) -> PLAY / PAUSE (SPACEBAR)
                elif current_gesture == "left_click" and confidence > 80:
                    if curr_time - last_action_time > ACTION_COOLDOWN:
                        pyautogui.press('space')
                        last_action_text = "EXECUTE: PLAY / PAUSE (Spacebar)"
                        last_action_time = curr_time
                        print("⏯️ Executed: PLAY / PAUSE (Spacebar)")

                # 5. SCROLL UP -> VOLUME UP
                elif current_gesture == "scroll_up" and confidence > 70:
                    if curr_time - last_volume_time > VOLUME_COOLDOWN:
                        pyautogui.press('volumeup')
                        last_action_text = "EXECUTE: VOLUME UP (+5%)"
                        last_volume_time = curr_time
                        print("🔊 Executed: VOLUME UP")

                # 6. SCROLL DOWN -> VOLUME DOWN
                elif current_gesture == "scroll_down" and confidence > 70:
                    if curr_time - last_volume_time > VOLUME_COOLDOWN:
                        pyautogui.press('volumedown')
                        last_action_text = "EXECUTE: VOLUME DOWN (-5%)"
                        last_volume_time = curr_time
                        print("🔉 Executed: VOLUME DOWN")

    else:
        frame_buffer.clear()

    # Performance Overlay
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Draw HUD Banner
    cv2.rectangle(frame, (0, 0), (cam_w, 110), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"AI Gesture: {current_gesture.upper()} ({confidence:.1f}%)", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {last_action_text}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    preview = cv2.resize(frame, (640, 480))
    cv2.imshow("Touchless AI OS Controller (Path 1)", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Touchless controller closed.")
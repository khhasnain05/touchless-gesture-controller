import cv2
import os
import csv
import time
import mediapipe as mp
from datetime import datetime

# ==================== SETTINGS ==================== #
SAVE_DIR = os.path.join("..", "data", "custom_landmarks")
SEQUENCE_LENGTH = 30          # frames per sample (about 1 second at ~30 FPS)
COUNTDOWN_SECONDS = 2         # delay before recording starts
# ================================================== #

os.makedirs(SAVE_DIR, exist_ok=True)

GESTURES = {
    "0": "idle",
    "1": "point",
    "2": "left_click",
    "3": "right_click",
    "4": "scroll_up",
    "5": "scroll_down",
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Return flat list: x,y,z for all 21 landmarks"""
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data

def save_sequence(sequence, label):
    """Save one recorded sequence as CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.csv"
    path = os.path.join(SAVE_DIR, filename)

    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["label"])
        for frame_landmarks in sequence:
            writer.writerow(frame_landmarks + [label])

    print(f"[SAVED] {path}  ({len(sequence)} frames)")

print("==============================================")
print(" AI Air Mouse - Phase 4 Dataset Collector")
print("==============================================")
print("Keys:")
for k, v in GESTURES.items():
    print(f"  Press {k} -> record '{v}'")
print("  Press q -> quit")
print("==============================================")

recording = False
countdown_active = False
countdown_start = 0
selected_label = None
sequence = []
prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    cam_h, cam_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_present = False
    landmark_row = None

    if result.multi_hand_landmarks:
        hand_present = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_row = extract_landmarks(hand_landmarks)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Countdown before recording
    if countdown_active:
        elapsed = current_time - countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed
        if remaining > 0:
            cv2.putText(
                frame,
                f"Recording '{selected_label}' in {int(remaining) + 1}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )
        else:
            countdown_active = False
            recording = True
            sequence = []
            print(f"[RECORDING] {selected_label} ... hold the gesture steady")

    # Active recording
    if recording:
        if hand_present and landmark_row is not None:
            sequence.append(landmark_row)
            cv2.putText(
                frame,
                f"RECORDING {selected_label}: {len(sequence)}/{SEQUENCE_LENGTH}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if len(sequence) >= SEQUENCE_LENGTH:
                save_sequence(sequence, selected_label)
                recording = False
                selected_label = None
                sequence = []
        else:
            cv2.putText(
                frame,
                "Hand not visible - recording paused",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # UI overlay
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "0:idle 1:point 2:left_click 3:right_click 4:scroll_up 5:scroll_down | q:quit",
        (10, cam_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    status = "READY"
    if countdown_active:
        status = "COUNTDOWN"
    elif recording:
        status = "RECORDING"
    cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.imshow("AI Air Mouse - Phase 4 Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Start countdown for selected gesture
    if (not recording) and (not countdown_active):
        key_char = chr(key) if key != 255 else ""
        if key_char in GESTURES:
            selected_label = GESTURES[key_char]
            countdown_active = True
            countdown_start = time.time()
            print(f"[READY] Get into '{selected_label}' pose...")

cap.release()
cv2.destroyAllWindows()
print("Phase 4 collector closed.")
print(f"Data saved in: {os.path.abspath(SAVE_DIR)}")
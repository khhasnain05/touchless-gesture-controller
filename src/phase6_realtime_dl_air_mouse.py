import os
import cv2
import time
import math
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque

# ---------------- Windows smooth cursor backend ----------------
try:
    import win32api
    import win32con
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()


def set_cursor(x, y):
    """Smoother on Windows than pyautogui alone."""
    x = int(max(0, min(SCREEN_W - 1, x)))
    y = int(max(0, min(SCREEN_H - 1, y)))
    if HAS_WIN32:
        win32api.SetCursorPos((x, y))
    else:
        pyautogui.moveTo(x, y)


# ============================================================
# One Euro Filter (Casiez et al.) - standard HCI realtime filter
# ============================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = t
            return float(x)

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ============================================================
# DL model (used for discrete gestures, NOT continuous move)
# ============================================================
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
model = MLPModel()
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {os.path.abspath(MODEL_PATH)}")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Loaded DL model for discrete gesture events.")

GESTURES = {
    0: "idle",
    1: "point",
    2: "left_click",
    3: "right_click",
    4: "scroll_up",
    5: "scroll_down",
}


def normalize_landmarks_hybrid(sequence_np):
    out = []
    for frame in sequence_np:
        lm = frame.reshape(21, 3)
        wrist = lm[0].copy()
        centered = lm - wrist
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        shape = centered / max_dist if max_dist > 0 else centered
        out.append(np.hstack((shape.flatten(), wrist)))
    return np.array(out, dtype=np.float32)


def finger_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y


def dist_px(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ============================================================
# Control tuning (start with these)
# ============================================================
# Camera region mapped to full screen (absolute + clutch)
MARGIN_X = 100
MARGIN_Y = 80

# One Euro Filter:
# lower min_cutoff = smoother but more lag
# higher beta = faster response during quick moves
FILTER_MIN_CUTOFF = 0.8
FILTER_BETA = 0.05

# Click
PINCH_ENTER = 35          # start pinch (pixels)
PINCH_EXIT = 50           # release pinch (hysteresis)
CLICK_COOLDOWN = 0.35
DOUBLE_CLICK_WINDOW = 0.65

# Scroll cooldown so it doesn't flood
SCROLL_COOLDOWN = 0.12

# Preview
PREVIEW_W, PREVIEW_H = 420, 320


# ============================================================
# Runtime state
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,          # faster/stabler on CPU
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fx = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
fy = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)

frame_buffer = deque(maxlen=30)

pinch_active = False
last_click_t = 0.0
last_scroll_t = 0.0
locked_x, locked_y = pyautogui.position()
prev_t = 0.0
mode = "REST"

print("=================================================")
print(" AI AIR MOUSE - STABLE CONTROL BUILD")
print(" Movement : geometry + One Euro Filter")
print(" Clicks   : pinch + target lock")
print(" DL model : event assist (click/scroll labels)")
print("=================================================")
print("HOW TO USE:")
print("1) Rest forearm on desk")
print("2) Index UP + other fingers DOWN = MOVE")
print("3) Open palm / fist = REST (cursor freezes)")
print("4) Pinch index+thumb = LEFT CLICK (cursor locks)")
print("5) Pinch twice quickly = DOUBLE CLICK")
print("Press q to quit")
print("=================================================")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    dl_gesture = "n/a"
    dl_conf = 0.0
    mode = "REST"

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # pixel points
        idx_tip = (int(lm.landmark[8].x * w), int(lm.landmark[8].y * h))
        th_tip = (int(lm.landmark[4].x * w), int(lm.landmark[4].y * h))
        mid_tip = (int(lm.landmark[12].x * w), int(lm.landmark[12].y * h))

        # geometry finger states
        index_is_up = finger_up(lm.landmark, 8, 6)
        middle_is_up = finger_up(lm.landmark, 12, 10)
        ring_is_up = finger_up(lm.landmark, 16, 14)
        pinky_is_up = finger_up(lm.landmark, 20, 18)

        pinch_d = dist_px(idx_tip, th_tip)

        # feed DL buffer (for discrete events / research overlay)
        raw = []
        for p in lm.landmark:
            raw.extend([p.x, p.y, p.z])
        frame_buffer.append(raw)

        if len(frame_buffer) == 30:
            seq = normalize_landmarks_hybrid(np.array(frame_buffer, dtype=np.float32))
            xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = torch.softmax(model(xt), dim=1)[0]
                conf, pred = torch.max(prob, 0)
                dl_gesture = GESTURES[int(pred)]
                dl_conf = float(conf) * 100.0

        # ---------------- PINCH CLICK (priority, target lock) ----------------
        if not pinch_active and pinch_d <= PINCH_ENTER:
            pinch_active = True
            # lock where cursor currently is
            locked_x, locked_y = win32api.GetCursorPos() if HAS_WIN32 else pyautogui.position()
            set_cursor(locked_x, locked_y)

            dt = now - last_click_t
            if dt >= CLICK_COOLDOWN:
                if dt <= DOUBLE_CLICK_WINDOW:
                    if HAS_WIN32:
                        # reliable double click via pyautogui event
                        pyautogui.doubleClick(locked_x, locked_y)
                    else:
                        pyautogui.doubleClick(locked_x, locked_y)
                    print("DOUBLE CLICK")
                else:
                    pyautogui.click(locked_x, locked_y)
                    print("LEFT CLICK")
                last_click_t = now
            mode = "LEFT_CLICK"
            fx.reset(); fy.reset()

        elif pinch_active and pinch_d >= PINCH_EXIT:
            pinch_active = False

        # ---------------- MOVE MODE (geometry only, continuous) ----------------
        elif (not pinch_active) and index_is_up and (not middle_is_up):
            mode = "MOVE"

            # absolute map index tip -> screen
            raw_x = np.interp(idx_tip[0], [MARGIN_X, w - MARGIN_X], [0, SCREEN_W - 1])
            raw_y = np.interp(idx_tip[1], [MARGIN_Y, h - MARGIN_Y], [0, SCREEN_H - 1])

            # One Euro Filter = smooth + low lag
            sm_x = fx(raw_x, now)
            sm_y = fy(raw_y, now)

            set_cursor(sm_x, sm_y)
            locked_x, locked_y = sm_x, sm_y
            cv2.circle(frame, idx_tip, 10, (0, 255, 255), -1)

        # ---------------- SCROLL (two fingers) ----------------
        elif (not pinch_active) and index_is_up and middle_is_up and (not ring_is_up) and (not pinky_is_up):
            mode = "SCROLL"
            fx.reset(); fy.reset()
            if now - last_scroll_t >= SCROLL_COOLDOWN:
                # top half = up, bottom half = down (simple + reliable)
                if idx_tip[1] < h * 0.45:
                    pyautogui.scroll(40)
                    mode = "SCROLL_UP"
                elif idx_tip[1] > h * 0.55:
                    pyautogui.scroll(-40)
                    mode = "SCROLL_DOWN"
                last_scroll_t = now

        # ---------------- RIGHT CLICK assist (thumb near middle) ----------------
        elif (not pinch_active) and dist_px(mid_tip, th_tip) <= PINCH_ENTER:
            mode = "RIGHT_CLICK"
            fx.reset(); fy.reset()
            if now - last_click_t >= CLICK_COOLDOWN:
                pyautogui.rightClick(locked_x, locked_y)
                print("RIGHT CLICK")
                last_click_t = now

        else:
            # REST / clutch
            mode = "REST"
            fx.reset(); fy.reset()

        # optional DL event overlay assist (does not drive continuous move)
        if dl_gesture in ("scroll_up", "scroll_down") and dl_conf > 85:
            # only mild assist if geometry already in scroll-ish pose
            pass

    else:
        frame_buffer.clear()
        fx.reset(); fy.reset()
        pinch_active = False
        mode = "NO_HAND"

    # ---------------- HUD ----------------
    fps = 1.0 / (now - prev_t) if prev_t else 0.0
    prev_t = now

    cv2.rectangle(frame, (MARGIN_X, MARGIN_Y), (w - MARGIN_X, h - MARGIN_Y), (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Control: {mode}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"DL Event: {dl_gesture} ({dl_conf:.0f}%)", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(frame, "MOVE=index only | REST=open palm | CLICK=pinch", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
    cv2.imshow("AI Air Mouse - Stable Build", preview)
    try:
        cv2.moveWindow("AI Air Mouse - Stable Build", SCREEN_W - PREVIEW_W - 30, 40)
    except Exception:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
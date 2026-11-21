import os
import cv2
import time
import torch
import torch.nn as nn
import pyautogui
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
from collections import deque
import threading

# PyAutoGUI Safety Setup
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# CustomTkinter High-End Dark Theme Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# -------------------------------------------------------------
# 1. PYTORCH MODEL ARCHITECTURE
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
# 2. PROFESSIONAL DASHBOARD APPLICATION
# -------------------------------------------------------------
class ModernTouchlessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Touchless Control Center - AI Gesture Recognition")
        self.geometry("1000 x 650")
        self.configure(fg_color="#0f172a") # Slate Dark Background
        self.resizable(False, False)

        # PyTorch Model Setup
        self.model = MLPModel()
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
            self.model.eval()
            self.model_status = "MLP Model Active (92.86% Acc)"
        else:
            self.model_status = "Model Not Found!"

        # Engine & Camera State
        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=30)
        
        # EDGE-TRIGGERED LATCH STATE (Fixes Double-Spacebar & Volume Spam)
        self.active_latch = None  # Stores current latched gesture
        self.last_volume_step_time = 0

        # Build UI
        self._build_dashboard()

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

    def _build_dashboard(self):
        # Header Bar
        self.header_frame = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=0, height=60)
        self.header_frame.pack(fill="x", side="top")
        
        ctk.CTkLabel(
            self.header_frame, 
            text=" ⚡ AI TOUCHLESS CONTROL CENTER", 
            font=("Segoe UI", 18, "bold"), 
            text_color="#f8fafc"
        ).pack(side="left", padx=20, pady=15)

        self.lbl_badge = ctk.CTkLabel(
            self.header_frame, 
            text="SYSTEM STANDBY", 
            font=("Segoe UI", 11, "bold"), 
            fg_color="#334155", 
            text_color="#94a3b8", 
            corner_radius=8,
            padx=12,
            pady=4
        )
        self.lbl_badge.pack(side="right", padx=20)

        # Main Workspace Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # LEFT CARD: High-Res Camera Preview
        self.cam_card = ctk.CTkFrame(self.main_container, fg_color="#1e293b", corner_radius=16)
        self.cam_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = ctk.CTkLabel(
            self.cam_card, 
            text="📷\nCamera Offline\nClick 'Start AI Engine' Below", 
            font=("Segoe UI", 16), 
            text_color="#64748b"
        )
        self.video_label.pack(expand=True, fill="both", padx=15, pady=15)

        # RIGHT CARD: Control Panel & Telemetry
        self.right_card = ctk.CTkFrame(self.main_container, fg_color="#1e293b", corner_radius=16, width=340)
        self.right_card.pack(side="right", fill="both", padx=(10, 0))
        self.right_card.pack_propagate(False)

        # Toggle Button
        self.btn_toggle = ctk.CTkButton(
            self.right_card, 
            text="Start AI Engine", 
            font=("Segoe UI", 14, "bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=48,
            corner_radius=12,
            command=self.toggle_engine
        )
        self.btn_toggle.pack(padx=20, pady=(20, 15), fill="x")

        # Telemetry Card
        self.telemetry_card = ctk.CTkFrame(self.right_card, fg_color="#0f172a", corner_radius=12)
        self.telemetry_card.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(self.telemetry_card, text="REAL-TIME TELEMETRY", font=("Segoe UI", 12, "bold"), text_color="#38bdf8").pack(anchor="w", padx=15, pady=(12, 8))

        self.lbl_fps = ctk.CTkLabel(self.telemetry_card, text="FPS Performance : --", font=("Segoe UI", 12), text_color="#cbd5e1")
        self.lbl_fps.pack(anchor="w", padx=15, pady=3)

        self.lbl_pose = ctk.CTkLabel(self.telemetry_card, text="AI Hand Pose     : STANDBY", font=("Segoe UI", 12, "bold"), text_color="#f59e0b")
        self.lbl_pose.pack(anchor="w", padx=15, pady=3)

        # Confidence Bar
        ctk.CTkLabel(self.telemetry_card, text="Confidence Score :", font=("Segoe UI", 12), text_color="#cbd5e1").pack(anchor="w", padx=15, pady=(6, 2))
        self.progress_conf = ctk.CTkProgressBar(self.telemetry_card, height=10, corner_radius=5, progress_color="#10b981")
        self.progress_conf.pack(fill="x", padx=15, pady=(0, 12))
        self.progress_conf.set(0)

        # Executed Action Card
        self.action_card = ctk.CTkFrame(self.right_card, fg_color="#0f172a", corner_radius=12)
        self.action_card.pack(padx=20, pady=10, fill="both", expand=True)

        ctk.CTkLabel(self.action_card, text="LAST EXECUTED ACTION", font=("Segoe UI", 12, "bold"), text_color="#38bdf8").pack(anchor="w", padx=15, pady=(12, 5))

        self.lbl_action = ctk.CTkLabel(
            self.action_card, 
            text="[ READY ]\nShow hand gesture", 
            font=("Segoe UI", 13, "bold"), 
            text_color="#34d399",
            justify="center"
        )
        self.lbl_action.pack(expand=True)

        # Model Footer Label
        ctk.CTkLabel(self.right_card, text=self.model_status, font=("Segoe UI", 10), text_color="#64748b").pack(pady=10)

    def toggle_engine(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return
            self.is_running = True
            self.btn_toggle.configure(text="Stop AI Engine", fg_color="#ef4444", hover_color="#dc2626")
            self.lbl_badge.configure(text="SYSTEM ACTIVE", fg_color="#064e3b", text_color="#34d399")
            
            threading.Thread(target=self.video_loop, daemon=True).start()
        else:
            self.is_running = False
            self.btn_toggle.configure(text="Start AI Engine", fg_color="#2563eb", hover_color="#1d4ed8")
            self.lbl_badge.configure(text="SYSTEM STANDBY", fg_color="#334155", text_color="#94a3b8")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="📷\nCamera Offline\nClick 'Start AI Engine' Below")

    def video_loop(self):
        prev_time = 0
        while self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            cam_h, cam_w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            current_gesture = "idle"
            confidence = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    raw_frame = []
                    for lm in hand_landmarks.landmark:
                        raw_frame.extend([lm.x, lm.y, lm.z])
                    self.frame_buffer.append(raw_frame)

                    if len(self.frame_buffer) == 30:
                        buf_np = np.array(self.frame_buffer, dtype=np.float32)
                        norm_seq = normalize_landmarks_hybrid(buf_np)
                        tensor_input = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            outputs = self.model(tensor_input)
                            probs = torch.softmax(outputs, dim=1)
                            conf, pred_class = torch.max(probs, 1)
                            current_gesture = GESTURE_LABELS[pred_class.item()]
                            confidence = conf.item() * 100

                        # Dispatch OS action with Edge-Trigger Latch
                        self.dispatch_action_latched(current_gesture, confidence)

            else:
                self.frame_buffer.clear()
                self.active_latch = None # Reset latch when hand leaves frame

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Update GUI Labels
            self.lbl_fps.configure(text=f"FPS Performance : {int(fps)}")
            self.lbl_pose.configure(text=f"AI Hand Pose     : {current_gesture.upper()}")
            self.progress_conf.set(confidence / 100.0)

            # Render Frame in GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img.resize((590, 480)))
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

    def dispatch_action_latched(self, gesture, conf):
        """
        EDGE-TRIGGERED STATE MACHINE:
        Guarantees 1 Pinch = EXACTLY 1 Spacebar Press (No Bouncing!).
        Action fires ONCE when entering gesture, and requires hand to release before firing again.
        """
        curr_time = time.time()

        # REST / IDLE: Clears Latch so gestures can fire again
        if gesture == "idle" or conf < 70:
            self.active_latch = None
            return

        # 1. PLAY / PAUSE (👌 PINCH) -> FIRES EXACTLY ONCE PER PINCH
        if gesture == "left_click" and conf >= 75:
            if self.active_latch != "left_click":
                pyautogui.press('space')
                self.lbl_action.configure(text="▶️ PLAY / PAUSE\n(Spacebar Fired)")
                self.active_latch = "left_click" # Lock latch!

        # 2. NEXT SLIDE / NEXT TRACK (👉 POINT) -> FIRES EXACTLY ONCE
        elif gesture == "point" and conf >= 75:
            if self.active_latch != "point":
                pyautogui.press('right')
                self.lbl_action.configure(text="⏩ NEXT TRACK / SLIDE\n(Right Arrow Fired)")
                self.active_latch = "point"

        # 3. PREVIOUS SLIDE / TRACK (✌️ PEACE) -> FIRES EXACTLY ONCE
        elif gesture == "right_click" and conf >= 75:
            if self.active_latch != "right_click":
                pyautogui.press('left')
                self.lbl_action.configure(text="⏪ PREV TRACK / SLIDE\n(Left Arrow Fired)")
                self.active_latch = "right_click"

        # 4. VOLUME UP (STEPPED QUANTIZATION)
        elif gesture == "scroll_up" and conf >= 70:
            if curr_time - self.last_volume_step_time > 0.25:
                pyautogui.press('volumeup')
                self.lbl_action.configure(text="🔊 VOLUME UP (+5%)")
                self.last_volume_step_time = curr_time

        # 5. VOLUME DOWN (STEPPED QUANTIZATION)
        elif gesture == "scroll_down" and conf >= 70:
            if curr_time - self.last_volume_step_time > 0.25:
                pyautogui.press('volumedown')
                self.lbl_action.configure(text="🔉 VOLUME DOWN (-5%)")
                self.last_volume_step_time = curr_time

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ModernTouchlessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
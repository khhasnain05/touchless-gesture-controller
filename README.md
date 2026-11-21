# ⚡ Real-Time AI Touchless OS Controller

An advanced, deep learning-powered Human-Computer Interaction (HCI) desktop application that enables touchless control of media playback, presentation navigation, and system volume using webcam hand gesture recognition.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)

</div>

---

## 🌟 Key Features

* **Real-Time Landmark Extraction:** 21-point 3D hand skeleton tracking via MediaPipe at 30+ FPS.
* **Hybrid Feature Normalization:** 66-feature spatial-temporal vector representation invariant to distance and hand size.
* **Trained PyTorch Neural Network:** Evaluated across MLP, 1D-CNN, and LSTM architectures (92.86% accuracy).
* **Edge-Triggered State Machine:** Eliminates command bouncing and duplicate shortcut triggers.
* **Modern Dark-Mode Dashboard:** Live telemetry, confidence meters, and embedded HD webcam feed built with CustomTkinter.

---

## 📊 Deep Learning Benchmark Comparison

| Architecture | Test Accuracy | Inference Latency | Status |
| :--- | :--- | :--- | :--- |
| **MLP (Selected)** | **92.86%** | **0.004 ms** | **Active Production Model** |
| 1D-CNN | 87.50% | 0.0253 ms | Evaluated |
| LSTM | 80.36% | 0.0520 ms | Evaluated |

---

## 📁 Repository Structure

```text
touchless-gesture-controller/
├── data/
│   ├── custom_landmarks/      # Raw CSV gesture recordings
│   └── processed/             # Normalized dataset (.npz)
├── docs/                      # Research paper & empirical usability study
├── models/                    # Exported PyTorch weights (.pth)
├── notebooks/                 # Google Colab training & benchmark notebook
├── src/
│   ├── phase1_hand_tracking.py
│   ├── phase2_cursor_control.py
│   ├── phase3_rule_gestures.py
│   ├── phase4_dataset_collector.py
│   ├── verify_dataset.py
│   ├── phase5_preprocess_dataset.py
│   ├── phase6_touchless_controller.py
│   └── app_gui.py             # Main Desktop Application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone & Setup Environment**

```bash
git clone https://github.com/khhasnain05/touchless-gesture-controller.git
cd touchless-gesture-controller
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run Desktop Application**

```bash
python src/app_gui.py
```

---

## 🖐️ Gesture Quick Reference

| Gesture | Hand Pose | Action |
| :--- | :--- | :--- |
| ✋ **`idle`** | Open Palm | **Standby / Rest** (No action) |
| 👌 **`left_click`** | Index + Thumb Pinch | **Play / Pause Media** (Spacebar) |
| 👉 **`point`** | Index Pointing | **Next Slide / Skip 5s** (Right Arrow) |
| ✌️ **`right_click`** | Middle + Thumb Pinch | **Previous Slide / Rewind 5s** (Left Arrow) |
| ⬆️ **`scroll_up`** | Peace Sign (Upper Screen) | **Volume Up** (+5%) |
| ⬇️ **`scroll_down`** | Peace Sign (Lower Screen) | **Volume Down** (-5%) |

> **Note:** Show an Open Palm (✋) briefly between gestures to reset the trigger cleanly.

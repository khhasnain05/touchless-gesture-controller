# ⚡ Real-Time AI Touchless OS Controller

An advanced, deep learning-powered Human-Computer Interaction (HCI) desktop application that enables touchless control of media playback, presentation navigation, and system volume using webcam hand gesture recognition.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)

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
ai-air-mouse/
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
│   ├── preprocess_dataset.py
│   ├── touchless_controller.py
│   └── app_gui.py             # Main Desktop Application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone & Setup Environment**

```bash
git clone [https://github.com/khhasnain05/ai-air-mouse.git](https://github.com/khhasnain05/ai-air-mouse.git)
cd ai-air-mouse
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run Desktop Application**

```bash
python src/app_gui.py
```

---

## 🖐️ Gesture Command Reference

* **✋ idle** : System Standby / Rest
* **👌 left_click** : Play / Pause Media (Spacebar)
* **👉 point** : Next Slide / Skip Track (Right Arrow)
* **✌️ right_click** : Previous Slide / Rewind Track (Left Arrow)
* **⬆️ ✌️ scroll_up** : Volume Up (+5%)
* **⬇️ ✌️ scroll_down** : Volume Down (-5%)
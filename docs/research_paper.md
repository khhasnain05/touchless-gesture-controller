# Real-Time Touchless Human-Computer Interaction via Hand Landmark Sequences and Deep Learning

## Abstract
This project presents a real-time, touchless Human-Computer Interaction (HCI) system that replaces traditional hardware input devices with vision-based hand gesture recognition. Using MediaPipe for 21-point hand landmark extraction, a custom 66-feature hybrid spatial-temporal normalization scheme was developed to ensure position-invariant gesture representation while maintaining spatial context. Three deep learning architectures—Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (1D-CNN), and Long Short-Term Memory (LSTM)—were trained and evaluated on a custom dataset of 280 normalized sequence samples across 6 gesture classes. The MLP model achieved optimal performance with 92.86% test accuracy and an ultra-low inference latency of 0.004 ms per sample. An empirical usability study revealed that continuous pixel-pointing introduces significant sensor jitter and physical fatigue ("Gorilla Arm Syndrome"), prompting an ergonomic pivot to discrete gesture command dispatching with edge-triggered state latching.

## 1. Research Benchmark Results
| Architecture | Test Accuracy (%) | Inference Latency (ms) | Real-Time Usability |
| :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | **92.86%** | **0.0040 ms** | **Optimal (30+ FPS, Zero-Lag)** |
| 1D-CNN | 87.50% | 0.0253 ms | Good |
| LSTM (Recurrent) | 80.36% | 0.0520 ms | Moderate |

## 2. Key Research Findings
1. **Feature Representation:** Normalizing 21 hand landmarks relative to the wrist while preserving absolute wrist screen coordinates (66 features) solved class ambiguity between identical finger shapes (e.g., Peace sign vs. Scroll position).
2. **Model Efficiency:** For compact landmark vector sequences, shallow dense architectures (MLP) converged faster and generalized better than recurrent architectures (LSTM) while requiring 13x less inference time.
3. **Ergonomic State Latching:** Edge-triggered state latches prevented gesture bouncing (e.g., unintended duplicate Spacebar presses during media playback) and eliminated muscle strain.
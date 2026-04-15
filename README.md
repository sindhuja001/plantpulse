---
title: AI + Quantum Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 🌿 AI + Quantum Plant Disease Detection

This is a hybrid AI-Quantum application designed to detect plant diseases from leaf images. It uses a **Random Forest Classifier** for initial image recognition and a **Qiskit Quantum Circuit** to perform statistical verification of the plant's health status.

## 🚀 How it works
1. **AI Processing**: A Scikit-Learn model analyzes image features extracted via OpenCV.
2. **Quantum Verification**: The image's intensity is encoded into a quantum circuit.
3. **Hybrid Decision**: The results from both the AI model and the quantum circuit are combined to provide a high-confidence diagnosis.

## 🛠️ Setup on Hugging Face
1. Upload all files to this Space.
2. Add your `IBM_QUANTUM_TOKEN` in the **Settings > Variables and secrets** section.
3. The app will automatically build and launch!

## 📄 License
MIT

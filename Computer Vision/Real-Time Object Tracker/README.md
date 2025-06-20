# 🚀 Real-Time Object Tracking with YOLOv8 (CPU vs GPU vs TensorRT)

This project demonstrates real-time object detection and tracking using the YOLOv8 model combined with the `short` tracker (SORT-based), benchmarked across three modes:

- **CPU**
- **GPU (PyTorch CUDA)**
- **TensorRT (Optimized GPU Inference)**

---

## 🧠 Overview

We use the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for detection and the `short` tracking algorithm to assign unique IDs to objects across frames. The goal is to compare performance across:

- Standard CPU inference
- PyTorch GPU acceleration
- TensorRT accelerated inference

---

## 📦 Features

- 🔍 Real-time object detection with YOLOv8n
- 🧠 Multi-object tracking using `short` (SORT)
- 📊 Benchmarking FPS across inference backends
- 🎥 Real-time video display with bounding boxes and IDs
- ✅ Full compatibility with `.avi` or `.mp4` files

---

## 🖥️ Requirements

```bash
pip install ultralytics opencv-python numpy
pip install short
```

> 🧪 For GPU and TensorRT:

- NVIDIA GPU with CUDA
- PyTorch with GPU support
- TensorRT engine installed and properly configured

---

## ⚙️ How to Run

### 🔹 CPU Inference

```bash
python main.py --mode cpu
```

### 🔹 GPU Inference (CUDA)

```bash
python main.py --mode gpu
```

### 🔹 TensorRT Inference

```bash
python main.py --mode tensorrt
```

> All modes will display video output in real-time and print FPS benchmarks at the end.

---

## 🧪 Benchmark Results

| Mode     | FPS   | Speedup |
| -------- | ----- | ------- |
| CPU      | 20.30 | 1.00x   |
| GPU      | 33.00 | 1.63x   |
| TensorRT | 41.03 | 2.02x   |

> Tested on: **NVIDIA RTX 2050**, Video: `traffic.avi`, Model: `yolov8n.pt`,`yolov8n.engine`

---

## 🔍 Comparison Summary

| Feature         | CPU         | GPU (CUDA)    | TensorRT     |
| --------------- | ----------- | ------------- | ------------ |
| Inference Speed | 🐢 Slower   | 🚀 Fast       | ⚡ Optimized |
| Accuracy        | ✅ Same     | ✅ Same       | ✅ Same      |
| Complexity      | 🟢 Easiest  | 🟡 Medium     | 🔴 Advanced  |
| Best Use Case   | Low-end CPU | General usage | Production   |

---

## 🧑‍💻 Author

[SK8-infi](https://github.com/SK8-infi)

---

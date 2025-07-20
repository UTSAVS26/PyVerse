# Dynamic Hardware Resource Monitor with Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)

A cross-platform, Python-based system monitor that displays real-time hardware usage and predicts future CPU, GPU, and RAM usage for the next 60 seconds using machine learning. Like `htop`, but smarter — with graphs, forecasts, and an interactive UI.

---

## Features
- **Real-Time Monitoring:**
  - CPU usage (per-core and total)
  - RAM and Swap memory
  - GPU load and VRAM usage (NVIDIA via pynvml)
  - Temperature monitoring (if available)
  - Battery status (if applicable)
- **Visual Dashboard:**
  - Live graphs for CPU, RAM, GPU
  - Temperature visualizations
  - Responsive dashboard (PyQt5 or Streamlit)
  - Export graphs as images
- **Forecasting with ML:**
  - Predicts next 60 seconds of CPU, GPU, RAM usage
  - LSTM (PyTorch) and ARIMA fallback
  - Real-time model training with sliding window
  - Graph overlays of predicted vs actual usage
- **Utilities:**
  - Logging and export to CSV
  - Configurable thresholds and update intervals
  - Easy packaging (PyInstaller)

---

## Architecture

```
hardware-monitor/
├── main.py                # Entry point
├── gui/
│   └── dashboard.py       # PyQt5/Streamlit dashboard
├── sensors/
│   ├── cpu.py             # CPU monitoring
│   ├── gpu.py             # GPU stats (NVIDIA)
│   └── memory.py          # RAM/Swap
├── models/
│   ├── lstm_predictor.py  # LSTM forecasting
│   └── stats_fallback.py  # ARIMA fallback
├── visualizations/
│   ├── live_plot.py       # Real-time graphs
│   └── prediction_overlay.py # Forecast overlays
├── utils/
│   ├── logger.py          # Logging
│   └── config.py          # Config/thresholds
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Run the Monitor**
```bash
python main.py --mode=qt    # PyQt5 GUI
python main.py --mode=web   # Streamlit dashboard
```

### 3. **Packaging (Optional)**
Create a portable executable:
```bash
pyinstaller --onefile main.py
```

---

## Sensors
- **CPU:** Uses `psutil` for per-core and total usage, and temperature (if available).
- **GPU:** Uses `pynvml` for NVIDIA GPU stats (utilization, VRAM, temperature).
- **Memory:** Uses `psutil` for RAM and swap.
- *(Planned: Disk, Network, Battery)*

## Models
- **LSTM Predictor:** PyTorch-based, fits and predicts 60s forward.
- **ARIMA Fallback:** Statsmodels-based, for environments without GPU/torch.

## Visualizations
- **Live Plot:** Real-time graphs (matplotlib/plotly).
- **Prediction Overlay:** Actual vs. predicted overlays.
- **Export:** Save graphs as PNG.

## Utilities
- **Logger:** Log metrics and export as CSV.
- **Config:** JSON-based, with thresholds and intervals.

---

## Configuration
Edit `utils/config.py` or use the `Config` class to set:
- `cpu_threshold`, `ram_threshold`, `gpu_threshold`
- `update_interval` (seconds)
- `prediction_interval` (seconds)

---

## Testing
Run all tests:
```bash
python -m pytest test_project.py
```
All core modules are covered by tests.

---

## Troubleshooting
- **No GPU detected:**
  - GPU stats will show an error if no NVIDIA GPU is present or `pynvml` is missing.
- **No temperature:**
  - Not all platforms support temperature sensors via `psutil`.
- **Matplotlib errors:**
  - For headless/test environments, the backend is set to `'Agg'` in tests.
- **PyQt5/Streamlit not installed:**
  - Install with `pip install pyqt5 streamlit`.

---

## FAQ
- **Can I add more sensors?**
  - Yes! Add modules in `sensors/` and wire them into the dashboard.
- **Can I use a different ML model?**
  - Yes, just implement a compatible class in `models/`.
- **Does it work on Linux/Mac?**
  - Yes, but some features (e.g., GPU, temperature) may vary by platform.

---

## 👤 Author
**Shivansh Katiyar**  
GitHub: [SK8-infi](https://github.com/SK8-infi)

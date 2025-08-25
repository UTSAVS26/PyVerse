# 🛰️ SignalSeekerAI – Adaptive Radio Noise Explorer for Cognitive Spectrum Learning

SignalSeekerAI is an AI system that simulates a noisy radio spectrum and learns to identify usable signals hidden in interference. The project is inspired by cognitive radios in IoT/5G networks, where devices must dynamically adapt to changing conditions and avoid congested frequencies.

## 🌟 Features

- **Procedural Spectrum Simulation**: No dataset required — the radio spectrum and noise are simulated procedurally
- **Multiple Signal Types**: Support for carrier tones, AM/FM modulated signals, and digital modulations
- **Adaptive Interference**: Dynamic interference sources that change over time
- **Multiple Agent Types**: 
  - Random scanning (baseline)
  - Sequential scanning
  - Adaptive scanning (power-based)
  - Deep Q-Network (RL agent)
- **Real-time Visualization**: Live spectrum monitoring with heatmaps and spectrograms
- **Multi-agent Competition**: Several radios competing for best channels
- **Comprehensive Testing**: Full test suite for all components

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SignalSeekerAI.git
cd SignalSeekerAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Run the training script**:
```bash
python train.py
```

2. **Create visualizations**:
```bash
python visualize.py
```

3. **Run tests**:
```bash
python -m pytest tests/
```

## 📁 Project Structure

```
SignalSeekerAI/
├── spectrum.py          # Radio spectrum simulation
├── agent.py            # Scanning/detection agents
├── train.py            # RL training loop
├── visualize.py        # Spectrum visualization
├── tests/              # Test suite
│   ├── test_spectrum.py
│   ├── test_agent.py
│   ├── test_training.py
│   └── test_visualization.py
├── models/             # Trained models (generated)
├── results/            # Output files (generated)
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🔧 Core Components

### 1. Radio Spectrum Simulation (`spectrum.py`)

The `RadioSpectrum` class simulates a dynamic radio environment:

```python
from spectrum import RadioSpectrum, SignalDetector

# Initialize spectrum
spectrum = RadioSpectrum(
    freq_range=(1e6, 100e6),  # 1-100 MHz
    num_channels=1000,
    noise_floor=-90  # dBm
)

# Get current spectrum
current_spectrum = spectrum.step()

# Detect signals
detector = SignalDetector(threshold=-70)
detected_signals = detector.detect_signals(current_spectrum, spectrum.frequencies)
```

**Features:**
- Procedural signal generation (carrier, AM, FM, digital)
- Dynamic interference sources
- Thermal noise simulation
- Time-varying spectrum conditions

### 2. Agent System (`agent.py`)

Multiple agent types for spectrum scanning:

```python
from agent import RandomAgent, SequentialAgent, AdaptiveAgent, DQNAgent

# Initialize different agents
random_agent = RandomAgent(num_channels=1000)
sequential_agent = SequentialAgent(num_channels=1000)
adaptive_agent = AdaptiveAgent(num_channels=1000)
dqn_agent = DQNAgent(num_channels=1000)

# Perform scan
scan_result = agent.scan_spectrum(spectrum, frequencies)
```

**Agent Types:**
- **RandomAgent**: Baseline random scanning
- **SequentialAgent**: Systematic frequency scanning
- **AdaptiveAgent**: Power-based adaptive scanning
- **DQNAgent**: Reinforcement learning agent

### 3. Training System (`train.py`)

Reinforcement learning training with custom reward design:

```python
from train import TrainingEnvironment, Trainer

# Initialize training environment
env = TrainingEnvironment(
    freq_range=(1e6, 100e6),
    num_channels=1000,
    detection_threshold=-70
)

# Train DQN agent
trainer = Trainer(env, dqn_agent)
training_stats = trainer.train(num_episodes=500)
```

**Reward System:**
- **+10.0**: Successful signal detection
- **-5.0**: False positive detection
- **-2.0**: Missed signal
- **-0.1**: Scanning cost
- **+5.0**: High-power signal bonus

### 4. Visualization (`visualize.py`)

Comprehensive visualization tools:

```python
from visualize import SpectrumVisualizer, TrainingVisualizer

# Spectrum visualization
visualizer = SpectrumVisualizer(spectrum, detector)
visualizer.plot_spectrum(current_spectrum, "Current Spectrum")
visualizer.plot_spectrogram(spectrum_history, time_steps, "Spectrum Over Time")

# Training visualization
training_viz = TrainingVisualizer()
training_viz.plot_training_progress(training_stats)
training_viz.plot_agent_comparison(comparison_results)
```

**Visualization Types:**
- Real-time spectrum plots
- Spectrograms and heatmaps
- Agent scanning behavior
- Training progress curves
- Agent comparison charts

## 🧪 Testing

The project includes comprehensive tests for all components:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_spectrum.py
python -m pytest tests/test_agent.py
python -m pytest tests/test_training.py
python -m pytest tests/test_visualization.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 📊 Performance Metrics

The system evaluates agents based on:

- **Detection Accuracy**: Percentage of real signals detected
- **False Positive Rate**: Incorrect signal detections
- **Coverage**: Percentage of spectrum scanned
- **Adaptability**: Performance under changing conditions
- **Efficiency**: Reward per time step

## 🔬 Research Applications

SignalSeekerAI is designed for research in:

- **Cognitive Radio**: Dynamic spectrum access
- **IoT Networks**: Spectrum sharing and interference avoidance
- **5G/6G**: Millimeter wave and dynamic spectrum allocation
- **Signal Processing**: Adaptive detection algorithms
- **Reinforcement Learning**: Multi-agent learning in wireless environments

## 🚀 Advanced Features

### Multi-Agent Competition

```python
from agent import MultiAgentEnvironment

# Create competitive environment
env = MultiAgentEnvironment(num_agents=4, num_channels=1000)
results = env.step(spectrum, frequencies)
```

### Custom Signal Types

```python
# Add custom signal parameters
spectrum.signal_params['custom'] = {
    'power': -35,
    'bandwidth': 15e3,
    'duration': 0.8,
    'modulation': 'custom'
}
```

### Real-time Dashboard

```python
from visualize import create_dashboard

# Create live monitoring dashboard
create_dashboard(spectrum, agent, num_steps=200)
```


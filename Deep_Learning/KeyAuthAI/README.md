# KeyAuthAI: Keystroke Dynamics-Based User Authentication

A Python-based behavioral biometrics system that authenticates users based on their unique typing patterns. KeyAuthAI analyzes keystroke dynamics including dwell time, flight time, and typing rhythm to create a secure authentication system.

## 🎯 Use Cases

- **Multi-factor Authentication**: Add keystroke dynamics as an additional security layer
- **Continuous Authentication**: Monitor user identity during active sessions
- **Access Control**: Secure access to sensitive applications and systems
- **Fraud Detection**: Identify unauthorized access attempts
- **Research Platform**: Study behavioral biometrics and machine learning

## 📁 Project Structure

```
KeyAuthAI/
├── data/
│   └── keystroke_logger.py      # Data collection and logging
├── features/
│   └── extractor.py             # Feature extraction from keystroke data
├── model/
│   ├── train_model.py           # Model training and management
│   └── verify_user.py           # User verification and authentication
├── ui/
│   └── auth_terminal.py         # Command-line interface
├── utils/
│   └── metrics.py               # Performance metrics and visualization
├── tests/                       # Comprehensive test suite
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 📊 Key Metrics

- **FAR (False Acceptance Rate)**: Rate of impostors incorrectly accepted
- **FRR (False Rejection Rate)**: Rate of legitimate users incorrectly rejected
- **EER (Equal Error Rate)**: Point where FAR equals FRR
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Accuracy**: Overall classification accuracy
- **Cross-validation**: Model robustness assessment

## 🚀 Features

### Core Functionality
- **Real-time Keystroke Capture**: Live monitoring of typing patterns
- **Feature Extraction**: 20+ behavioral features including:
  - Dwell time statistics (mean, std, percentiles)
  - Flight time analysis
  - N-gram timing patterns
  - Statistical moments (skewness, kurtosis)
  - Rhythm consistency metrics
- **Machine Learning Models**:
  - **Supervised**: SVM, Random Forest, KNN
  - **Unsupervised**: One-Class SVM, Isolation Forest
  - **Advanced**: LSTM for temporal sequences (optional)

### User Experience
- **Simple Registration**: Type passphrase 3-5 times to build profile
- **Quick Authentication**: Single passphrase entry for verification
- **Interactive CLI**: Color-coded terminal interface
- **Progress Tracking**: Real-time feedback during data collection

### Security & Performance
- **Threshold-based Authentication**: Configurable security levels
- **Model Persistence**: Save/load trained models
- **Batch Processing**: Handle multiple sessions efficiently
- **Error Handling**: Graceful failure recovery

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/KeyAuthAI.git
   cd KeyAuthAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -m pytest tests/ -v
   ```

## 📖 Usage

### Quick Start

1. **Register a new user**:
   ```bash
   python -m keyauthai.ui.auth_terminal
   # Follow prompts to register with username and passphrase
   ```

2. **Authenticate existing user**:
   ```bash
   # Use the same interface to authenticate
   ```

### Programmatic Usage

```python
from keyauthai.data.keystroke_logger import KeystrokeLogger
from keyauthai.features.extractor import FeatureExtractor
from keyauthai.model.train_model import KeystrokeModelTrainer
from keyauthai.model.verify_user import UserVerifier

# 1. Collect training data
logger = KeystrokeLogger()
logger.start_recording("alice", "my_secret_passphrase")
# ... user types passphrase ...
session_data = logger.stop_recording()

# 2. Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(session_data)

# 3. Train model
trainer = KeystrokeModelTrainer()
results = trainer.train_model("alice", "svm", min_sessions=3)

# 4. Verify user
verifier = UserVerifier()
is_authentic = verifier.verify_user("alice", session_data)
```

### Advanced Configuration

```python
# Custom thresholds
verifier.set_threshold("svm", 0.8)

# Model comparison
models = verifier.list_available_models("alice")
for model_type in models:
    accuracy = verifier.get_user_stats("alice")[model_type]['accuracy']
    print(f"{model_type}: {accuracy:.3f}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_data_collection.py -v
python -m pytest tests/test_model_training.py -v
python -m pytest tests/test_ui_components.py -v

# Generate coverage report
python -m pytest tests/ --cov=keyauthai --cov-report=html
```

## 📈 Performance Evaluation

```python
from keyauthai.utils.metrics import KeystrokeMetrics

# Calculate performance metrics
metrics = KeystrokeMetrics()
far, frr = metrics.calculate_far_frr(genuine_scores, impostor_scores, threshold)
eer, optimal_threshold = metrics.calculate_eer(genuine_scores, impostor_scores)

# Generate visualizations
metrics.plot_roc_curve(genuine_scores, impostor_scores)
metrics.plot_score_distributions(genuine_scores, impostor_scores)
```

## 🔧 Configuration

### Model Parameters

```python
# Supervised models
svm_params = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Unsupervised models
one_class_svm_params = {
    'nu': 0.1,
    'kernel': 'rbf'
}
```

### Feature Selection

```python
# Custom feature sets
timing_features = ['total_time', 'avg_interval', 'typing_speed_cps']
dwell_features = ['avg_dwell_time', 'std_dwell_time', 'dwell_cv']
flight_features = ['avg_flight_time', 'std_flight_time', 'flight_cv']
```

## 🚨 Security Considerations

- **Data Privacy**: Keystroke data is stored locally by default
- **Encryption**: Consider encrypting stored models and data
- **Threshold Tuning**: Balance security vs. usability
- **Model Validation**: Regular retraining with new data
- **Anomaly Detection**: Monitor for unusual patterns

---

**Author**: @SK8-infi 
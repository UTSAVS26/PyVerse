# 🔐 PassClass: AI-Powered Password Strength Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance](#performance)
- [Author](#author)

## 🎯 Overview

**PassClass** is an intelligent password strength evaluation system that uses machine learning to automatically classify passwords as weak, medium, or strong. The system generates synthetic training data and employs multiple ML algorithms to provide accurate password strength predictions.

### Key Highlights

- **🤖 ML-Powered**: Uses TF-IDF features with Random Forest, SVM, and Logistic Regression
- **📊 Self-Generated Data**: Creates synthetic password datasets with automatic labeling
- **🎯 High Accuracy**: Achieves 94%+ accuracy on test data
- **🌐 Web Interface**: Interactive Streamlit app for real-time testing
- **📈 Comprehensive Evaluation**: Detailed metrics and visualizations

## ✨ Features

### Core Functionality
- **Password Generation**: Synthetic password creation with varying complexity levels
- **Automatic Labeling**: Rule-based heuristics for strength classification
- **ML Classification**: Multiple algorithms for robust predictions
- **Confidence Scoring**: Probability-based predictions with confidence levels
- **Detailed Analysis**: Character-by-character breakdown of password strength

### Technical Features
- **TF-IDF Vectorization**: Character-level n-gram features (1-3 grams)
- **Multiple Models**: Random Forest, SVM, and Logistic Regression
- **Cross-Validation**: Robust model evaluation with 5-fold CV
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Visualization**: Confusion matrices, feature importance, and performance plots

### User Interface
- **Streamlit Web App**: Interactive password strength tester
- **Real-time Analysis**: Instant feedback on password strength
- **Visual Indicators**: Color-coded strength levels and confidence bars
- **Detailed Reports**: Comprehensive analysis with improvement suggestions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, streamlit; print('Installation successful!')"
   ```

## ⚡ Quick Start

### 1. Generate Training Data
```bash
python data/generate_passwords.py
```
This creates a balanced dataset of 3,000 passwords (1,000 per strength level).

### 2. Train the Model
```bash
python training/train_model.py
```
Trains multiple models and saves the best performing one.

### 3. Run the Web App
```bash
streamlit run ui/streamlit_app.py
```
Opens an interactive web interface for password testing.

### 4. Evaluate Performance
```bash
python training/evaluate.py
```
Generates detailed performance reports and visualizations.

## 📖 Usage

### Command Line Usage

#### Password Generation
```python
from data.generate_passwords import PasswordGenerator

generator = PasswordGenerator()
df = generator.generate_dataset(total_samples=3000)
print(f"Generated {len(df)} passwords")
```

#### Password Labeling
```python
from labeling.labeler import PasswordLabeler

labeler = PasswordLabeler()
password = "MySecurePass123!"
label = labeler.label_password(password)
analysis = labeler.get_detailed_analysis(password)
print(f"Password: {password}")
print(f"Strength: {label}")
print(f"Issues: {analysis['issues']}")
```

#### Model Prediction
```python
from models.tfidf_classifier import TFIDFPasswordClassifier

classifier = TFIDFPasswordClassifier()
classifier.load_model('models/random_forest_password_classifier.pkl')

result = classifier.predict_with_confidence("TestPassword123!")
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Web Interface

The Streamlit app provides an intuitive interface for password testing:

1. **Input Field**: Enter any password for analysis
2. **Real-time Results**: Instant strength classification
3. **Detailed Breakdown**: Character analysis and improvement suggestions
4. **Confidence Metrics**: Probability scores for each strength level
5. **Visual Feedback**: Color-coded strength indicators

## 🏗️ Architecture

### Project Structure
```
passclass/
├── data/
│   ├── generate_passwords.py       # Password generation engine
│   └── password_dataset.csv        # Generated training data
├── labeling/
│   └── labeler.py                  # Rule-based labeling system
├── models/
│   └── tfidf_classifier.py         # ML classifier implementation
├── training/
│   ├── train_model.py              # Model training pipeline
│   └── evaluate.py                 # Performance evaluation
├── ui/
│   └── streamlit_app.py            # Web interface
├── utils/
│   └── metrics.py                  # Visualization utilities
├── tests/
│   └── test_generation.py          # Test suite
├── requirements.txt
├── README.md
└── LICENSE
```

### Data Flow

1. **Generation**: Synthetic passwords created with controlled complexity
2. **Labeling**: Rule-based classification using security heuristics
3. **Training**: TF-IDF features + ML algorithms for pattern learning
4. **Prediction**: Real-time classification with confidence scores
5. **Evaluation**: Comprehensive performance analysis

### Model Pipeline

```
Password Input → TF-IDF Vectorization → ML Classification → Confidence Scoring → Output
```

## 📚 API Reference

### PasswordLabeler

The core labeling engine with rule-based classification.

#### Methods

- `label_password(password: str) -> str`: Classify password strength
- `get_detailed_analysis(password: str) -> dict`: Comprehensive analysis
- `calculate_strength_score(password: str) -> dict`: Character metrics
- `is_common_password(password: str) -> bool`: Check against weak passwords
- `contains_common_words(password: str) -> bool`: Detect common patterns

#### Example
```python
labeler = PasswordLabeler()
analysis = labeler.get_detailed_analysis("MyPass123!")
print(analysis['label'])  # 'medium'
print(analysis['issues'])  # ['No special characters']
```

### TFIDFPasswordClassifier

Machine learning classifier with TF-IDF features.

#### Methods

- `train(df: pd.DataFrame) -> dict`: Train the model
- `predict(password: str) -> str`: Get prediction
- `predict_with_confidence(password: str) -> dict`: Prediction with confidence
- `save_model(filepath: str)`: Save trained model
- `load_model(filepath: str)`: Load saved model

#### Example
```python
classifier = TFIDFPasswordClassifier(model_type='random_forest')
classifier.train(training_data)
result = classifier.predict_with_confidence("SecurePass123!")
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### PasswordGenerator

Synthetic password generation with controlled complexity.

#### Methods

- `generate_weak_password() -> str`: Generate weak passwords
- `generate_medium_password() -> str`: Generate medium passwords
- `generate_strong_password() -> str`: Generate strong passwords
- `generate_dataset(total_samples: int) -> pd.DataFrame`: Create training dataset

#### Example
```python
generator = PasswordGenerator()
weak_pwd = generator.generate_weak_password()
medium_pwd = generator.generate_medium_password()
strong_pwd = generator.generate_strong_password()
```

## 💡 Examples

### Basic Usage

```python
# Quick password strength check
from labeling.labeler import PasswordLabeler

labeler = PasswordLabeler()
passwords = ["abc", "password123", "Hello2023!", "G7^s9L!zB1m"]

for pwd in passwords:
    label = labeler.label_password(pwd)
    print(f"{pwd:15} -> {label}")
```

### Advanced Analysis

```python
# Detailed password analysis
from labeling.labeler import PasswordLabeler

labeler = PasswordLabeler()
password = "MySecurePass123!"

analysis = labeler.get_detailed_analysis(password)
print(f"Password: {analysis['password']}")
print(f"Strength: {analysis['label'].upper()}")
print(f"Length: {analysis['score']['length']}")
print(f"Character Types: {analysis['score']['uppercase']} uppercase, "
      f"{analysis['score']['lowercase']} lowercase, "
      f"{analysis['score']['digits']} digits, "
      f"{analysis['score']['special']} special")
print(f"Issues: {analysis['issues']}")
print(f"Strengths: {analysis['strengths']}")
```

### Model Training

```python
# Complete training pipeline
from data.generate_passwords import PasswordGenerator
from models.tfidf_classifier import TFIDFPasswordClassifier

# Generate data
generator = PasswordGenerator()
df = generator.generate_dataset(total_samples=3000)

# Train model
classifier = TFIDFPasswordClassifier(model_type='random_forest')
results = classifier.train(df)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Cross-validation: {results['cv_mean']:.2%} (+/- {results['cv_std']*2:.2%})")

# Save model
classifier.save_model('my_password_classifier.pkl')
```

### Web Application

```python
# Run the Streamlit app
import streamlit as st
from models.tfidf_classifier import TFIDFPasswordClassifier

# Load model
classifier = TFIDFPasswordClassifier()
classifier.load_model('models/random_forest_password_classifier.pkl')

# Streamlit interface
st.title("🔐 Password Strength Classifier")
password = st.text_input("Enter password:", type="password")

if password:
    result = classifier.predict_with_confidence(password)
    st.write(f"**Strength:** {result['predicted_label'].upper()}")
    st.write(f"**Confidence:** {result['confidence']:.1%}")
```

## 📊 Performance

### Evaluation Metrics

- **Overall Accuracy**: 94.23%
- **Mean ROC AUC**: 98.80%
- **Per-Class Performance**:
  - Weak: 92.44% precision, 92.90% recall
  - Medium: 91.98% precision, 90.60% recall
  - Strong: 98.22% precision, 99.20% recall
### Test Results

Example password classification accuracy: **82.4%** (14/17 correct)

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Tests
```bash
# Test password labeling
python -m pytest tests/test_generation.py::TestPasswordLabeler -v

# Test password generation
python -m pytest tests/test_generation.py::TestPasswordGenerator -v

# Test integration
python -m pytest tests/test_generation.py::TestIntegration -v
```

### Test Coverage
- ✅ Password labeling accuracy
- ✅ Password generation quality
- ✅ Model training pipeline
- ✅ Data consistency checks
- ✅ Performance benchmarks

## 🔧 Configuration

### Model Parameters

```python
# TF-IDF Vectorizer
vectorizer_params = {
    'analyzer': 'char',
    'ngram_range': (1, 3),
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.95
}

# Random Forest
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}
```

### Labeling Rules

The system uses composite scoring based on:
- **Length**: 0-25 points (6-12+ characters)
- **Character Variety**: 0-35 points (uppercase, lowercase, digits, special)
- **Uniqueness**: 0-20 points (character diversity ratio)
- **Pattern Penalties**: Deductions for repeating/sequential patterns

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Add tests for new functionality
4. **Run tests**: `python -m pytest tests/ -v`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/passclass.git
cd passclass
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Generate data and train
python data/generate_passwords.py
python training/train_model.py

# Start development server
streamlit run ui/streamlit_app.py
```

### Code Style

- Follow PEP 8 guidelines
- Add type hints for all functions
- Include docstrings for all classes and methods
- Write comprehensive tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**SK8-infi**

- GitHub: [@SK8-infi](https://github.com/SK8-infi)

---

<div align="center">

**Made with ❤️ by SK8-infi**

[![GitHub](https://img.shields.io/badge/GitHub-SK8--infi-black.svg?style=flat&logo=github)](https://github.com/SK8-infi)

</div> 
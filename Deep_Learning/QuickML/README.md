# 📊 QuickML – Mini AutoML Engine

A powerful yet simple AutoML engine that automatically processes any CSV dataset and finds the best machine learning model.

## 🚀 Features

- **Universal CSV Support**: Works with any classification or regression dataset
- **Automatic Preprocessing**: Handles missing values, encoding, and scaling
- **Multi-Model Training**: Tests 5 different algorithms automatically
- **Smart Model Selection**: Picks the best performing model
- **Beautiful UI**: Streamlit-based interface with interactive visualizations
- **Feature Importance**: Explains model decisions with importance plots
- **Model Export**: Saves the best model for later use

## 🛠 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QuickML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python quickml.py --data your_dataset.csv --target target_column
```

## 📊 Supported Algorithms

- **Classification**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting
- **Regression**: Linear Regression, Random Forest, SVM, KNN, Gradient Boosting

## 📈 Example Output

```
Best Model: RandomForestClassifier
Accuracy: 0.92
Cross-validation Score: 0.89
Feature Importance: [0.25, 0.18, 0.15, ...]
Model saved to: best_model.pkl
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=quickml --cov-report=html
```

## 📁 Project Structure

```
QuickML/
├── quickml/
│   ├── __init__.py
│   ├── core.py          # Main AutoML engine
│   ├── preprocessing.py # Data preprocessing utilities
│   ├── models.py        # Model definitions and training
│   ├── evaluation.py    # Model evaluation and metrics
│   └── visualization.py # Plotting and visualization
├── app.py               # Streamlit web interface
├── quickml.py           # Command line interface
├── tests/               # Test suite
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

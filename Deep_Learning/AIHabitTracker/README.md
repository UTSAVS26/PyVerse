# AI Habit Tracker with Pattern Detection

A comprehensive habit tracking application that uses machine learning to detect patterns and correlations between your daily habits and productivity/mood.

## 🚀 Features

- **Daily Habit Logging**: Track sleep, exercise, screen time, water intake, and work/study hours
- **Mood & Productivity Tracking**: Rate your daily mood and productivity (1-5 scale)
- **Pattern Detection**: ML-powered analysis to find correlations between habits and outcomes
- **Predictive Insights**: Get warnings about potential low-performance days
- **Visual Analytics**: Beautiful charts and heatmaps showing your patterns
- **100% Offline**: All data stays on your machine
- **Adaptive Learning**: Improves predictions as you collect more data

## 📊 Sample Data Structure

| Date       | Sleep (hrs) | Exercise (min) | Screen Time (hrs) | Water (glasses) | Work Hours | Mood (1-5) | Productivity (1-5) |
|------------|-------------|----------------|-------------------|-----------------|------------|------------|-------------------|
| 2025-01-01 | 7.5         | 30             | 4                 | 8               | 8          | 4          | 4                 |
| 2025-01-02 | 6.0         | 0              | 8                 | 4               | 6          | 2          | 2                 |

## 🛠 Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### GUI Application
```bash
python src/gui_app.py
```

### Streamlit Dashboard
```bash
streamlit run src/streamlit_app.py
```

### Command Line Interface
```bash
python src/cli_app.py
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

## 📁 Project Structure

```
AIHabitTracker/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── habit_model.py
│   │   └── database.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pattern_detector.py
│   │   └── visualizer.py
│   ├── gui_app.py
│   ├── streamlit_app.py
│   └── cli_app.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_analysis.py
│   └── test_integration.py
├── data/
│   └── habits.db
├── requirements.txt
└── README.md
```

## 🔬 ML Features

- **Correlation Analysis**: Find relationships between habits and outcomes
- **Time Series Prediction**: Predict future productivity based on patterns
- **Anomaly Detection**: Identify unusual days that break your patterns
- **Recommendation Engine**: Suggest optimal habit combinations

## 📈 Example Insights

- "Your productivity peaks after 7.5+ hours of sleep"
- "Exercise days show 40% higher mood scores"
- "Screen time over 6 hours correlates with 2-point mood drops"
- "Warning: 3+ days without exercise predicts low productivity"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

# 🧩 CodeSage – AI-Enhanced Code Complexity Estimator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-85%20passed-brightgreen.svg)](https://github.com/yourusername/codesage)

**CodeSage** is an intelligent Python tool that analyzes code complexity using Abstract Syntax Tree (AST) analysis and machine learning to provide actionable insights for code quality improvement.

## ✨ Features

### 🔍 **Core Analysis**
- **AST-Based Parsing**: Deep code structure analysis using Python's built-in `ast` module
- **Static Metrics**: Cyclomatic complexity, maintainability index, function length, nesting depth
- **Multi-File Support**: Analyze entire projects or individual Python files
- **Smart Filtering**: Automatically ignores common directories (`.git`, `__pycache__`, `venv`, etc.)

### 🤖 **AI/ML Enhancements**
- **Anomaly Detection**: Uses `IsolationForest` to identify unusual code patterns
- **Clustering Analysis**: `KMeans` clustering to group similar complexity patterns
- **Pattern Recognition**: `TfidfVectorizer` for analyzing function naming conventions
- **AI Risk Scoring**: Combines ML anomaly scores with rule-based metrics for comprehensive risk assessment

### 📊 **Rich Reporting**
- **Beautiful CLI**: Rich terminal output with progress indicators and color-coded metrics
- **Interactive HTML**: Plotly-powered dashboards with charts and visualizations
- **Actionable Insights**: Specific suggestions for code improvement
- **Risk Hotspots**: Identifies problematic code areas with detailed analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codesage.git
cd codesage

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Analyze a single file
python -m codesage.cli examples/sample_code.py

# Analyze entire project
python -m codesage.cli .

# Generate HTML report
python -m codesage.cli . --html

# Detailed analysis with all options
python -m codesage.cli . --html --detailed --html-output custom_report.html
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `path` | File or directory to analyze | Required |
| `--html` | Generate HTML report | False |
| `--html-output` | HTML output filename | `codesage_report.html` |
| `--detailed` | Show detailed function analysis | False |
| `--no-ml` | Disable machine learning features | False |
| `--quiet` | Suppress progress output | False |
| `--verbose` | Show verbose output | False |
| `--strict` | Exit with error on high-risk code | False |
| `--version` | Show version and exit | - |
| `--help` | Show help message | - |

## 🔧 AI/ML Features Deep Dive

### Anomaly Detection
CodeSage uses **Isolation Forest** to identify code that deviates from normal patterns:

```python
# The system automatically trains on your codebase
# Functions with unusual complexity patterns get higher anomaly scores
# This helps identify code that might need refactoring
```

### Complexity Clustering
**K-Means clustering** groups functions by complexity characteristics:

```python
# Functions are grouped into clusters based on:
# - Cyclomatic complexity
# - Lines of code
# - Nesting depth
# - Parameter count
# - Return statement count
```

### Pattern Recognition
**TF-IDF vectorization** analyzes function naming patterns:

```python
# Identifies naming conventions
# Detects potential code smells
# Suggests improvements based on patterns
```

## 📊 Sample Output

### CLI Report
```
🧩 CodeSage Analysis Report
===============================================================================

📊 Project Overview
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Metric              ┃ Value    ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ Total Files         │ 12       │ 📁     │
│ Total Lines         │ 4692     │ 📝     │
│ Total Functions     │ 155      │ ⚙️      │
│ Avg Complexity      │ 5.0      │ ✅     │
│ Avg Maintainability │ 93.3/100 │ ✅     │
│ AI Risk Level       │ LOW      │ ✅     │
└─────────────────────┴──────────┴────────┘

🤖 AI Insights
  1. 📏 AI identified 3 functions with high line count (avg: 101.3 lines).
     Consider extracting helper methods.
  2. 📈 High variance in file sizes detected. Consider standardizing module sizes.

💡 Suggestions for Improvement
  1. Function 'very_complex_algorithm': Consider breaking this function into smaller, focused functions
  2. Function 'deeply_nested_function': Use early returns to reduce nesting depth
```

### HTML Dashboard
- **Interactive Charts**: Complexity distribution, maintainability trends, risk hotspots
- **Detailed Metrics**: Function-by-function breakdown with risk scores
- **Exportable Reports**: Save and share analysis results

## 🏗️ Architecture

```
codesage/
├── __init__.py          # Package initialization
├── analyzer.py          # Main analysis engine
├── metrics.py           # Complexity calculations & ML models
├── reporter.py          # CLI & HTML report generation
├── cli.py              # Command-line interface
└── tests/              # Comprehensive test suite
    ├── test_analyzer.py
    ├── test_metrics.py
    ├── test_reporter.py
    └── test_cli.py
```

## 🧪 Testing

The project includes comprehensive testing with **85 test cases** covering all functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=codesage --cov-report=html

# Run specific test file
python -m pytest tests/test_metrics.py -v
```

## 📦 Dependencies

### Core Dependencies
- **`ast`**: Python's built-in Abstract Syntax Tree module
- **`radon`**: Code complexity metrics calculation
- **`pandas` & `numpy`**: Data manipulation and numerical operations
- **`matplotlib` & `plotly`**: Data visualization and HTML charts

### AI/ML Dependencies
- **`scikit-learn`**: Machine learning models (IsolationForest, KMeans, TfidfVectorizer)
- **`rich`**: Beautiful terminal output and progress bars

### Development Dependencies
- **`pytest`**: Testing framework
- **`black`**: Code formatting
- **`flake8`**: Linting

## 🎯 Use Cases

### For Developers
- **Code Review**: Identify complex functions before code review
- **Refactoring**: Find code that would benefit from restructuring
- **Quality Gates**: Set complexity thresholds in CI/CD pipelines

### For Teams
- **Knowledge Transfer**: Understand code complexity across the team
- **Technical Debt**: Track and prioritize refactoring efforts
- **Standards**: Enforce coding standards and best practices

### For Projects
- **Onboarding**: Help new developers understand codebase complexity
- **Documentation**: Generate complexity reports for stakeholders
- **Maintenance**: Identify areas that need attention

## 🚧 Requirements

- **Python**: 3.8 or higher
- **Memory**: 512MB RAM (for large projects)
- **Dependencies**: See `requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

*Empowering developers to write better, more maintainable code through AI-powered insights.*

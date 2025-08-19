# 🧹 PyPolish - AI Code Cleaner and Rewriter

**Domain:** NLP / Developer Tools

## 📌 What it does

PyPolish accepts raw Python scripts and transforms them into clean, optimized, and more Pythonic versions by:

- Detecting non-idiomatic code, poor formatting, redundant logic, and anti-patterns
- Rewriting code using PEP 8 formatting (`black`)
- Refactoring with AST-based analysis
- Applying Python best practices
- Outputting a clean, optimized version with before/after diffs

## ⚙️ How it works

1. **Parsing**: Load code into an AST using Python's `ast` module
2. **Analysis**: Identify unused variables/imports, long functions, nested loops, bad naming
3. **Rewriting**: Use AST transformations to rewrite logic without changing functionality
4. **Final Output**: Show cleaned code, before vs after diff, and refactoring suggestions

## 🛠 Tech Stack

- **Parsing & Transformation**: `ast`, `rope`
- **Linting**: `flake8`, `pylint`
- **Formatting**: `black`, `isort`
- **Diff Viewer**: `difflib` for before/after changes
- **UI**: CLI with rich formatting

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### CLI Usage

```bash
# Clean a single file
python -m pypolish clean input.py

# Clean a file and save to output
python -m pypolish clean input.py --output cleaned.py

# Show diff only
python -m pypolish diff input.py
```

### Example

**Before:**
```python
import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
```

**After:**
```python
import math

def calc(x: int) -> None:
    """Print whether x is even or odd."""
    print("Even" if x % 2 == 0 else "Odd")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=pypolish

# Run specific test file
pytest tests/test_code_cleaner.py
```

## 📁 Project Structure

```
pypolish/
├── pypolish/
│   ├── __init__.py
│   ├── cli.py
│   ├── code_cleaner.py
│   ├── ast_analyzer.py
│   ├── formatter.py
│   └── diff_viewer.py
├── tests/
│   ├── __init__.py
│   ├── test_code_cleaner.py
│   ├── test_ast_analyzer.py
│   ├── test_formatter.py
│   └── test_diff_viewer.py
├── requirements.txt
├── README.md
└── setup.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests to ensure everything works
6. Submit a pull request

## 📄 License

MIT License

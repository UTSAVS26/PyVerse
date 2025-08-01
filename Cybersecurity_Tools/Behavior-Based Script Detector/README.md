# 🛡️ Behavior-Based Script Detector

A static analysis tool for Python scripts that detects potentially malicious or risky behavior by inspecting the code structure and patterns — without executing it.

## 📌 Project Overview

Behavior-Based Script Detector is a Python-powered static code analyzer designed to evaluate .py scripts for suspicious or potentially harmful behavior. It uses Python's Abstract Syntax Tree (AST) to parse and analyze code structure, flagging risky operations like:

- Network activity (downloads, remote access)
- File system manipulation (deleting, modifying sensitive files)
- Process forking or shell execution
- Obfuscated or encoded code blocks

It assigns a behavioral risk score, highlights reasons for the score, and optionally suggests mitigation or safe alternatives.

## 🔍 Key Features

### 🧠 Static AST-Based Analysis
- No execution required
- Safe for scanning unknown/untrusted scripts

### 🚩 Detects Suspicious Patterns
- `os.system`, `subprocess`, `eval`, `exec`, `pickle`, etc.
- File writes to sensitive paths (`/etc`, `~/.ssh`)
- URL access / downloads
- Encoding/decoding functions (`base64`, `marshal`, etc.)

### 🧾 Risk Scoring System
- Each matched pattern contributes to an overall score
- Provides clear breakdown: which lines, which risks

### 📦 Open-Source Vetting Mode
- Designed to vet unverified Python scripts and tools before use
- Can be integrated into CI/CD workflows or Git hooks

### 💡 Explainable Output
- Easy-to-understand output with line numbers and reasons

## 🛠️ Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| AST Parsing | `ast`, `asttokens` |
| Risk Analysis | Custom ruleset |
| CLI Interface | `argparse`, `rich` |
| Optional TUI | `textual` |
| Report Export | `json`, `markdown`, `html` |

## 📂 Project Structure

```
behavior_script_detector/
├── analyzer/
│   ├── pattern_rules.py
│   ├── score_calculator.py
│   └── report_generator.py
├── examples/
│   └── test_malicious.py
├── cli/
│   └── main.py
├── reports/
│   ├── scan_results.json
│   └── flagged_code.md
├── utils/
│   └── file_loader.py
├── tests/
│   └── test_patterns.py
├── README.md
└── requirements.txt
```

## 🧪 Sample Usage

```bash
# Scan a script for suspicious behavior
python cli/main.py --file examples/test_malicious.py

# Output detailed report
python cli/main.py --file examples/test_malicious.py --report reports/

# Batch scan a directory of scripts
python cli/main.py --dir ./downloads/ --threshold 60
```

## 🔎 Sample Output

```json
{
  "filename": "script.py",
  "risk_score": 78,
  "verdict": "High Risk",
  "suspicious_patterns": [
    {
      "line": 12,
      "pattern": "exec usage",
      "description": "Dynamic code execution using exec()"
    },
    {
      "line": 24,
      "pattern": "network download",
      "description": "Attempts to download files from remote URL"
    },
    {
      "line": 41,
      "pattern": "subprocess",
      "description": "Runs shell command using subprocess"
    }
  ]
}
```

## 📈 Use Cases

- ✅ Vetting untrusted or third-party Python scripts before execution
- 🧪 Research and education on malware or obfuscation techniques
- 🔐 Adding to security pipelines (e.g., GitHub Actions or pre-commit hooks)
- 🧰 Building a secure Python sandbox or malware analysis lab

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd behavior-script-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest tests/ -v
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=analyzer --cov=utils --cov=cli
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## ⚠️ Disclaimer

This tool is for educational and security research purposes. Always use responsibly and in accordance with applicable laws and regulations. 
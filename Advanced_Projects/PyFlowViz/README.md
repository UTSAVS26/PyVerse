# PyFlowViz: Code-to-Flowchart Auto Generator

🔁 **PyFlowViz**: Instantly visualize Python logic with dynamic flowcharts.

## 📌 Project Overview

PyFlowViz is a smart developer tool that automatically parses Python code and generates flowcharts representing its logic structure — including conditionals, loops, and function calls.

The tool uses Python's built-in ast module or bytecode to analyze code, then maps the structure into flowchart diagrams that can be exported as static images (SVG/PNG) or interactive HTML pages.

### Ideal for:
- Code comprehension
- Debugging and teaching
- Documentation automation

## 🚀 Key Features

### 🔍 Code Parsing with AST or Bytecode
Supports parsing of any .py file using Python's Abstract Syntax Tree (ast) or low-level bytecode for deeper analysis.

### 🔁 Dynamic Flowchart Generation
Visualizes:
- If-else branches
- Loops (for, while)
- Function calls and returns
- Try-except blocks

### 🎨 Clean, Interactive Visuals
Output charts via:
- graphviz for SVG/PNG
- mermaid.js or d3.js for HTML

### 📂 Batch Mode
Process multiple Python files and export diagrams in one go.

### ✨ Code-to-Chart Sync
Hover or click to highlight corresponding lines in original source.

### 🖥️ Optional GUI
Lightweight GUI using PyQt5 or Gradio for non-terminal users.

## 🧩 Tech Stack

| Component | Technology |
|-----------|------------|
| Parser Engine | ast, inspect, bytecode |
| Visualization | graphviz, mermaid.js, pyvis |
| Frontend GUI | Gradio, PyQt5, or webview |
| Output Format | SVG, PNG, HTML |
| File Handling | Python standard I/O (os, pathlib) |

## 🛠 Sample Usage

```bash
$ pyflowviz example.py --output flow.svg

✔ Parsed: example.py
✔ Generated flowchart: flow.svg
```

```bash
$ pyflowviz app.py --html

✔ Generated interactive flowchart: flowchart.html
```

## 🧠 Supported Code Elements

- ✅ Functions & returns
- ✅ Conditionals: if / elif / else
- ✅ Loops: for, while
- ✅ Try/except blocks
- ✅ Function calls
- ✅ Nested structures

## 🔧 Installation

```bash
git clone https://github.com/yourusername/pyflowviz
cd pyflowviz
pip install -r requirements.txt
python pyflowviz.py
```

## 📂 Folder Structure

```
pyflowviz/
├── parser/
│   ├── ast_parser.py        # AST-based parser
│   ├── bytecode_parser.py   # Bytecode fallback (optional)
├── visualizer/
│   ├── graphviz_gen.py      # Graphviz diagram generation
│   ├── html_renderer.py     # Mermaid/D3 output
├── gui/
│   └── app_gui.py           # Optional GUI frontend
├── examples/
│   └── (sample python files)
├── pyflowviz.py             # CLI entry point
├── requirements.txt
└── README.md
```

## 🎛️ CLI Options

```
Usage: pyflowviz [file.py] [OPTIONS]

Options:
  --output [filename.svg|.png]   Save diagram as image
  --html                         Export as interactive HTML
  --batch folder/                Parse and export all Python files in folder
  --gui                          Launch optional GUI
```

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=pyflowviz tests/
```

## 📝 License

MIT License 
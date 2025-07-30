# 🧠 PySleuth: Context-Aware Python Debugger

## Project Overview
PySleuth is an advanced Python debugger that augments traditional debugging tools like pdb by giving rich context to code execution.

It shows not only:
- Current variable states
- Execution stack
- Line-by-line tracing
But also explains why specific branches (e.g., if, while, function calls) were taken — by tracking execution history, conditions, and data flow.

Inspired by tools like snoop, icecream, and VizTracer, but focused on contextual reasoning.

## Key Features
- **Why-Driven Debugging**: Explains why branches were taken
- **Runtime Execution Tracing**: Captures line-by-line execution using sys.settrace() or bytecode
- **AST-Based Annotation**: Uses ast to analyze conditions, variable assignments, and control flow
- **Smart Logging**: Automatically logs interesting events
- **TUI Debug Panel (Optional)**: A curses-based interface
- **Drop-in Decorator Mode**: Like @snoop, use @pysleuth to trace functions without clutter

## Tech Stack
| Component         | Technology                |
|-------------------|--------------------------|
| Code Analysis     | ast, inspect, dis, tokenize |
| Runtime Tracing   | sys.settrace, linecache, tracemalloc |
| TUI (Optional)    | curses, urwid, Textual    |
| Decorator Support | Python decorators         |
| Logging / Export  | JSON, plaintext, HTML     |

## Sample Usage
### 1. Decorator Mode
```python
from pysleuth import trace

@trace
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```
**Output**
```
✅ Called is_prime(n=7)
➡️ Line 3: if n <= 1 → False (n = 7)
🔁 for loop from 2 to 6
  ↳ Checked n % 2 == 0 → False
  ↳ Checked n % 3 == 0 → False
  ↳ Checked n % 4 == 0 → False
✅ Returned True
```

### 2. TUI Mode
```
$ pysleuth run myscript.py
```
Displays:
- Code view
- Branch reason log
- Variables at each step
- History of path decisions

## Folder Structure
```
pysleuth/
├── core/
│   ├── tracer.py          # sys.settrace logic
│   ├── explainer.py       # Branch reason extraction
│   ├── ast_parser.py      # Static code analysis
│   └── logger.py          # Log formatter
├── tui/
│   └── curses_ui.py       # Optional TUI frontend
├── decorators/
│   └── trace.py           # @trace decorator
├── examples/
│   └── sample_scripts.py
├── pysleuth.py            # CLI interface
├── requirements.txt
└── README.md
```

## Installation
```bash
git clone https://github.com/yourusername/pysleuth
cd pysleuth
pip install -r requirements.txt
```

## CLI Options
```
Usage: pysleuth run script.py [--tui] [--log output.json]

Options:
  --tui        Launch TUI mode
  --log        Export log to file (json/txt)
  --trace-fn   Trace only specific function names
```

## Planned Features
- Visual HTML-based code playback
- Export annotated source code (with inline explanations)
- Integration with VSCode or Jupyter Notebooks
- Real-time remote debugging over socket 
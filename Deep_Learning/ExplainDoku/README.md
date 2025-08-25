# 🧩 ExplainDoku — Sudoku Solver with Human-Style Explanations

**Domain:** Constraint Solving + NLP  
**What it does:** Solves user-provided Sudoku with classic strategies + backtracking, and generates **step-by-step, human-readable** explanations (e.g., "This 5 goes here because it's the only candidate in row 3").

## 🎯 Goals

* **Accurate solver** that finishes most valid 9×9 puzzles
* **Faithful explanations** aligned with the exact rule used at each step
* **Readable traces** and an interactive "Next step" UI

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Solve a puzzle with explanations
python -m explaindoku.cli solve --grid "530070000600195000098000060800060003400803001700020006060000280000419005000080079" --explain

# Interactive step-by-step solving
python -m explaindoku.cli step --file examples/medium.txt

# Web UI
streamlit run explaindoku/ui/streamlit_app.py
```

## 🗂️ Project Structure

```
explaindoku/
│
├─ core/
│  ├─ grid.py                # Board representation, parsing, formatting
│  ├─ constraints.py         # Peers, units, candidate sets, AC-3 helpers
│  ├─ strategies/
│  │  ├─ singles.py          # Naked/Hidden Singles
│  │  ├─ locked_candidates.py# Pointing/Claiming
│  │  ├─ pairs_triples.py    # Naked/Hidden Pairs/Triples
│  │  └─ fish.py             # X-Wing (stretch)
│  ├─ search.py              # MRV + backtracking; value ordering
│  └─ solver.py              # Orchestrates: apply rules -> fallback to search
│
├─ explain/
│  ├─ trace.py               # Structured proof trace (JSON)
│  ├─ templates.py           # Natural-language templates per technique
│  └─ verbalizer.py          # Turns trace steps into sentences
│
├─ io/
│  ├─ parse.py               # From string/CSV/SDK format
│  └─ export.py              # Export steps to HTML/PDF
│
├─ ui/
│  ├─ cli.py                 # Solve once or step-by-step in terminal
│  └─ streamlit_app.py       # Interactive web UI
│
├─ tests/
│  ├─ test_strategies.py
│  ├─ test_solver.py
│  └─ test_explanations.py
│
├─ examples/
│  └─ easy_medium_hard.txt
│
├─ README.md
├─ requirements.txt
└─ LICENSE
```

## 🧠 Solving Pipeline

1. **Constraint Initialization**
   * Build units & peers for each cell (rows, cols, boxes)
   * Maintain candidate sets; optionally run **AC-3** to prune arcs

2. **Human Techniques (in order of difficulty)**
   * **Naked Single** (only one candidate in a cell)
   * **Hidden Single** (only place for a digit in a unit)
   * **Locked Candidates** (pointing/claiming)
   * **Naked/Hidden Pairs/Triples**
   * **X-Wing** (stretch goal)
   * Each technique **emits a trace event** with all eliminations/placements

3. **Search Fallback**
   * **MRV** (Minimum Remaining Values) variable ordering
   * **Heuristic value ordering** via simple ML
   * Plain **backtracking** with explanation trace noting "assumption branches" and conflicts

4. **Trace → Explanation**
   * Every step is a typed record with technique, unit, digit, placement, and evidence
   * Natural language templates convert structured data to human-readable explanations

## 🗣️ Example Output

```
Step 1 — Naked Single:
R1C3 has only candidate 9 → Place 9 in R1C3.

Step 2 — Locked Candidates (pointing):
In Box (Row1-3, Col1-3), digit 7 appears only in Row 2 → eliminate 7 from R2C4, R2C6.

Step 3 — Hidden Single:
Only cell in Column 5 that can take 3 is R6C5 → Place 3 in R6C5.
```

## 🛠️ Tech Stack

- Python 3.8+
- numpy - Numerical operations
- dataclasses - Data structures
- rich - CLI formatting
- streamlit - Web UI
- scikit-learn - ML heuristics (optional)
- pytest - Testing

## 📊 Metrics & Validation

* **Functional:** Solve rate, average steps, backtracks
* **Explanation quality:** % of placements explained by human rule (vs search), template coverage, faithfulness check
* **Performance:** avg ms per strategy pass, memory

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_solver.py

# Run with coverage
pytest --cov=explaindoku tests/
```

## 📄 License

MIT License - see LICENSE file for details.

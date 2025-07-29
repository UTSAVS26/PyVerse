# SmartCLI: Natural Language Command-Line Toolkit

## Overview

SmartCLI is a Python-based command-line utility that allows users to interact with their operating system using natural language instructions. It translates user queries like:

    "Find and compress all PNGs in Downloads from June"

into safe, executable shell commands. SmartCLI leverages NLP models (local transformers or OpenAI API) to parse queries and map them to system operations, making the command line accessible to both technical and non-technical users.

## Features

- **Natural Language to Command Translation**: Parses and translates user queries into shell commands using NLP models (HuggingFace Transformers, OpenAI API, etc.).
- **Safety Layer**: All generated commands are sandboxed and previewed for user approval before execution. Dangerous commands (like `rm -rf`) are blocked by default.
- **Semantic File Operations**: Understands queries such as:
    - "all PDFs from last month"
    - "large images in my desktop"
    - "zipped folders older than 1 week"
- **Context Awareness**: Recognizes system paths (Downloads, Desktop, Documents, etc.) and interprets them relative to the user's environment.
- **Extensible Command Library**: Modular backend allows for easy addition of new command types and integrations (e.g., git, ffmpeg, docker).
- **Command Preview & Logging**: Shows the interpreted command before execution and logs all actions with timestamp, user, and approval status.

## Tech Stack

| Component         | Technology                            |
| ----------------- | ------------------------------------- |
| Language Parsing  | HuggingFace Transformers / OpenAI API |
| CLI Interface     | argparse, rich (for styling)          |
| OS Interaction    | Python's subprocess, os, shutil       |
| Security          | Sandboxed command executor            |
| Logging           | Plaintext log file (smartcli.log)     |

## Installation

1. Clone the repository:

    git clone https://github.com/UTSAVS26/PyVerse
    cd PyVerse/Advanced_Projects/SmartCLI
    cd SmartCLI

2. Install dependencies:

    pip install -r requirements.txt

3. (Optional) For OpenAI API support, set your API key:

    export OPENAI_API_KEY=your_openai_key

4. Run SmartCLI:

    python smartcli.py "Delete all zip files in Downloads older than 1 month"

## Usage Examples

- **Delete old zip files:**

      python smartcli.py "Delete all zip files in Downloads older than 1 month"

      Parsed Command:
      > find ~/Downloads -name '*.zip' -mtime +30 -delete

- **List recent PDFs:**

      python smartcli.py "List all PDF files from last week on Desktop"

      Parsed Command:
      > find ~/Desktop -name '*.pdf' -mtime -7

- **Use OpenAI for parsing:**

      python smartcli.py "Compress large images in Documents" --openai

## Architecture

```
SmartCLI/
├── core/
│   ├── parser.py         # NLP query parsing (transformers/OpenAI/regex)
│   ├── command_map.py    # Maps parsed intents/entities to shell commands
│   ├── executor.py       # Executes commands with safety checks and approval
│   └── utils.py          # Path resolution, logging, and helpers
├── models/               # (Optional) Place for local LLMs
├── smartcli.py           # CLI entry point
├── requirements.txt      # Python dependencies
├── test_smartcli.py      # Automated test suite
└── README.md
```

## Extensibility

- **Add new command types:**
  - Extend `core/command_map.py` to support new actions or file operations.
- **Integrate new NLP models:**
  - Update `core/parser.py` to use additional or custom models.
- **Plug in external tools:**
  - Add modules for git, ffmpeg, docker, etc., and map new intents.

## Testing

A comprehensive test suite is provided in `test_smartcli.py`.

To run all tests:

    python -m unittest test_smartcli.py

Tests cover:
- NLP parsing and entity extraction
- Command mapping for various actions and file types
- Safety checks for dangerous commands
- Logging and utility functions
- Integration and edge cases
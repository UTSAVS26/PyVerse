# PyTerminus

🖥️ **PyTerminus: Virtual Multi-Terminal Manager in Python**

A powerful, Python-based terminal session manager — split, persist, search.

## 📌 Project Overview

PyTerminus is a terminal-based multi-shell session manager, built entirely in Python. Inspired by tools like tmux and screen, it allows users to run and manage multiple terminal sessions within a single CLI window using a flexible, modern TUI interface.

With support for split panes, custom shell environments, scrollback logging, and persistent sessions, PyTerminus is perfect for power users, sysadmins, or developers looking for a Python-native solution.

## 🚀 Key Features

- **🪟 Split Terminal Interface**: Horizontally or vertically split your terminal into multiple interactive shells
- **🔄 Persistent Sessions**: Powered by Python's pty and os.fork, resume work where you left off
- **🧠 Smart Command Logging**: Logs commands and outputs per session with timestamps and optional search
- **🕹️ TUI Navigation (Curses/Urwid)**: Switch, resize, rename panes via keyboard shortcuts in a responsive terminal UI
- **🔍 Searchable Scrollback**: Scroll through terminal history and search across outputs with regex or fuzzy match
- **🎯 Session Profiles**: Define and launch sessions with predefined commands (e.g., Python shell, SSH, docker, etc.)
- **🧩 Plugin Architecture (Planned)**: Add custom panes (e.g., system monitor, Git status, chat, etc.)

## 🧩 Tech Stack

| Component | Technology |
|-----------|------------|
| Terminal UI | urwid |
| Pseudo-Terminals | pty, os.fork, subprocess |
| Input Handling | select, termios, tty |
| Logging | JSON or plain text logs |
| Configurations | yaml, argparse, dataclasses |

## 🛠 Sample Workflow

```bash
$ pyterminus

🪟 [Pane 1]: zsh (~/projects)
🪟 [Pane 2]: python3
🪟 [Pane 3]: htop

⎈ Ctrl+n: new pane
⎈ Ctrl+arrows: move
⎈ Ctrl+s: save session
⎈ Ctrl+/: search history
```

```bash
$ pyterminus --session dev-work.yaml

✔ Loaded session: 3 terminals (web server, redis, docker shell)
```

## 🔧 Installation

```bash
git clone https://github.com/yourusername/pyterminus
cd pyterminus
pip install -r requirements.txt
python pyterminus.py
```

## 📂 Folder Structure

```
pyterminus/
├── tui/
│   ├── layout.py            # TUI layout manager
│   ├── keybindings.py       # Shortcut config
│   └── main_ui.py          # Main UI component
├── core/
│   ├── terminal_pane.py     # Pty + input/output piping
│   ├── session_manager.py   # Persistent session saving/loading
│   └── logger.py            # Smart log writer
├── profiles/
│   ├── sample_session.yaml  # Example profile
│   └── web_dev.yaml        # Web development profile
├── tests/
│   ├── test_session_manager.py
│   ├── test_logger.py
│   ├── test_keybindings.py
│   └── test_layout.py
├── pyterminus.py            # Main entry point
├── requirements.txt
└── README.md
```

## 🎛️ CLI Options

```bash
Usage: pyterminus [OPTIONS]

Options:
  --session [file.yaml]        Load saved session profile
  --log-dir [folder/]          Specify custom log directory
  --theme [dark|light]         UI theme
  --shell [bash|zsh|fish]      Default shell for new panes
  --debug                      Enable debug mode
```

## 📄 Example Session Profile

```yaml
name: dev-env
panes:
  - name: server
    command: python manage.py runserver
    cwd: ~/projects/myapp
  - name: redis
    command: redis-cli
  - name: shell
    command: zsh
```

## ⌨️ Key Bindings

### Navigation
- `Ctrl+n`: New pane
- `Ctrl+Tab`: Next pane
- `Ctrl+Shift+Tab`: Previous pane
- `Ctrl+1-9`: Switch to pane 1-9

### Pane Management
- `Ctrl+x`: Close pane
- `Ctrl+r`: Rename pane
- `Ctrl+s`: Save session
- `Ctrl+l`: Load session

### Layout
- `Ctrl+h`: Split horizontal
- `Ctrl+v`: Split vertical
- `Ctrl++`: Increase pane size
- `Ctrl+-`: Decrease pane size

### Search
- `Ctrl+/`: Search history
- `Ctrl+f`: Search output
- `Ctrl+g`: Clear search

### Help & Exit
- `F1`: Show help
- `F2`: Show status
- `F3`: Show logs
- `Ctrl+q`: Quit
- `Ctrl+\`: Force quit

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=tui --cov-report=html

# Run specific test file
pytest tests/test_session_manager.py
```

## 🐛 Known Issues

- **Windows Support**: Limited support due to pty module differences
- **Terminal Resizing**: May not work perfectly in all terminal emulators
- **Shell Integration**: Some advanced shell features may not work as expected

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Inspired by tmux and screen
- Built with urwid for TUI
- Uses Python's pty module for pseudo-terminals

## 📊 Project Status

- ✅ Core functionality implemented
- ✅ Session management working
- ✅ TUI interface functional
- ✅ Logging system complete
- ✅ Test suite comprehensive
- 🔄 Plugin architecture (planned)
- 🔄 Advanced search features (planned)
- 🔄 Windows compatibility (planned)

---

**Made with ❤️ by @SK8-infi** 
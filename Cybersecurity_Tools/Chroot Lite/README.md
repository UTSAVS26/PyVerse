# Chroot Lite 🔐

A lightweight, terminal-controlled sandbox system for securely executing untrusted Python code or CLI scripts within isolated environments using chroot, resource, and subprocess.

## 📌 Overview

Chroot Lite is a terminal-based sandbox manager that mimics lightweight containerization. It allows users to create isolated execution environments — useful for testing or securely running unknown scripts — without the overhead of full Docker containers or VMs.

Each sandbox:
- Runs in a separate fake root (chroot) directory
- Has CPU/memory limits
- Can block internet access
- Is managed through a simple CLI interface

## 🔧 Features

### 📁 Chroot-based Directory Isolation
- Emulate a fake root filesystem for each sandbox
- Prevent access to system files and binaries

### 🧠 Resource Limiting
- Limit CPU time, memory usage, file sizes using Python's resource module
- Terminate processes that exceed limits

### 🚫 Internet Access Blocking
- Optional blocking of outgoing connections using firewall rules or Python netfilter hooks

### 🐚 Secure Code Execution
- Run Python or shell scripts using subprocess inside a controlled environment
- Read-only mounts and safe script execution

### 🖥️ Terminal-Based CLI Manager
- Create, list, run, or destroy sandboxes via commands like:
  ```bash
  sandbox create mybox --limit-mem 128MB
  sandbox run mybox script.py
  ```

### 📄 Logging and Audit
- Log all commands and execution history
- Optional file output redirection and result capture

## 🛠 Tech Stack

| Component | Tool/Library |
|-----------|--------------|
| Environment isolation | `os.chroot()`, `os.fork()` |
| Resource limits | `resource`, `signal` |
| Execution engine | `subprocess`, `pty` |
| CLI Interface | `argparse`, `cmd`, `rich` |
| Logging/Auditing | `logging`, `datetime` |

## 📂 Project Structure

```
chroot-lite/
├── sandbox/
│   ├── manager.py         # Sandbox creation, deletion
│   ├── executor.py        # Code execution and resource limiting
│   ├── limiter.py         # Memory/CPU constraints
│   └── firewall.py        # Optional internet block
├── cli/
│   └── main.py            # Terminal interface
├── templates/
│   └── base_rootfs/       # Minimal fake root structure
├── examples/
│   └── test_script.py
├── logs/
│   └── sandbox_history.log
├── tests/
│   ├── test_manager.py
│   ├── test_executor.py
│   ├── test_limiter.py
│   └── test_integration.py
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/yourname/chroot-lite.git
cd chroot-lite
pip install -r requirements.txt
```

⚠️ **Note**: chroot requires root privileges on Linux/macOS.

### 🧪 Example Usage

```bash
# Create a new sandbox with memory limit
sudo python cli/main.py create mybox --memory 128 --cpu 20

# Run a Python script inside the sandbox
sudo python cli/main.py run mybox examples/test_script.py

# List available sandboxes
sudo python cli/main.py list

# Destroy the sandbox
sudo python cli/main.py delete mybox
```

## 🧠 How It Works

### Chroot Setup
- Copies a minimal base rootfs template (with /bin/python, /lib, etc.)
- Uses `os.chroot()` to change the root directory of subprocess

### Forked Execution
- Forks a child process and applies resource limits (`RLIMIT_AS`, `RLIMIT_CPU`)
- Launches the given script using `subprocess.run`

### Network Isolation
- Optionally uses iptables/firewall-cmd to block the PID's network access

### Logging & Audit
- Records execution metadata (command, resource usage, status) in logs

## 🌐 Limitations

- Works only on Unix-like systems with chroot support
- Requires sudo to create chroot jails
- Does not virtualize kernel syscalls like full containers (not a replacement for Docker)
- Should not be used as a full security boundary in production

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sandbox --cov-report=html

# Run specific test file
pytest tests/test_manager.py
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## ⚠️ Security Notice

This tool is for educational and testing purposes. It provides basic isolation but should not be considered a complete security solution for production environments. 
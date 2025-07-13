#  Secure Remote Shell Over WebSockets

A lightweight, cross-platform, AES-encrypted remote shell system over WebSockets, built with Python. Includes authentication and simple CLI/GUI.

---

##  Project Overview
Secure Remote Shell Over WebSockets is a secure, real-time, bidirectional command execution framework. It allows a client to connect to a server using WebSocket, authenticate with a token, and execute shell commands remotely in an AES-encrypted tunnel.

---

##  Features
-  **AES-256 Encryption** for secure communication
-  **Token-based Authentication**
-  **WebSocket-based Full-Duplex Communication**
-  **Remote Shell Execution** (Bash / PowerShell / CMD)
-  **Cross-Platform**: Linux, Windows, macOS
-  **Modular**: Easily extendable for scripting, logging, or file transfer
-  **CLI & Minimal GUI** (Optional PyQt/Tkinter UI)
-  **Firewall Friendly**: Works even through NAT if port is forwarded

---

##  Architecture
```
+-------------------+            Encrypted WebSocket Tunnel           +--------------------+
|    Client (CLI)   | <--------------------------------------------> |    Server (Shell)  |
| - Connect to IP   |                                               | - WebSocket Server |
| - Authenticate    |                                               | - Token Check      |
| - Send Commands   |                                               | - Execute Commands |
| - View Output     |                                               | - Return Output    |
+-------------------+                                               +--------------------+
```

---

##  Security
- **AES Encryption**: All messages are encrypted with a pre-shared AES-256 key (derived using PBKDF2 from password).
- **Authentication**: Token-based handshake before shell access is granted.
- **Replay Attack Prevention**: Nonce or timestamp system can be implemented for extra protection.

---

##  Project Structure
```
secure-shell-ws/
│
├── client/
│   ├── client.py           # CLI client
│   ├── gui_client.py       # Optional GUI version (PyQt5/Tkinter)
│   └── utils.py            # Encryption, key handling
│
├── server/
│   ├── server.py           # WebSocket server handling command exec
│   ├── shell_handler.py    # Secure shell command executor
│   └── auth.py             # Token auth & key derivation
│
├── config/
│   └── config.json         # Server/Client config (host, port, token, etc.)
│
├── tests/                  # Unit and integration tests
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  Sample Usage

###  Server
```bash
python server/server.py --port 8765 --token SECRET123 --password mypass
```

###  Client
```bash
python client/client.py --host 127.0.0.1 --port 8765 --token SECRET123 --password mypass
```

You can also run a single command and exit:
```bash
python client/client.py --host 127.0.0.1 --port 8765 --token SECRET123 --password mypass --command "echo Hello"
```

---

##  Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/secure-shell-ws.git
   cd secure-shell-ws
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Edit `config/config.json`** (optional):
   - Set your host, port, token, and password.

---

##  Technologies Used
- Python 3
- websockets (async WebSocket communication)
- cryptography (AES encryption)
- argparse, subprocess, json, base64, hashlib

---

##  Use Cases
- Remote administration of servers (through NAT/firewall)
- Secure IoT command execution
- Education or demonstration of encrypted shell access
- Custom DevOps scripting platform

---

##  Disclaimer
This tool is for educational and ethical use only. Unauthorized access to computer systems is illegal and punishable by law. Always have permission before connecting to a device.

---
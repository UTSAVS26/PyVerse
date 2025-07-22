# Multi-LAN Screen Share

## Multi-Device Screen Share via LAN

A high-performance, Python-powered tool to broadcast your screen in real time to multiple devices over a local network. Designed for classrooms, labs, Raspberry Pi displays, and air-gapped environments.

---

## Features

- **Low Latency**: Real-time screen sharing with minimal delay.
- **One-to-Many**: Broadcast to multiple receivers simultaneously.
- **Headless Support**: Works on servers and Raspberry Pi with no display.
- **UDP & TCP**: Choose between fast, lossy UDP or reliable TCP transport.
- **Multiple GUIs**: View on Tkinter or OpenCV-based clients.
- **Configurable**: Easily adjust quality, compression, and network settings.
- **Cross-Platform**: Runs on Windows, Linux, macOS, and Raspberry Pi.

---

## Project Structure

```
multi-lan-screen-share/
├── server/
│   ├── broadcaster_udp.py      # UDP stream server
│   ├── broadcaster_tcp.py      # TCP stream server
│   ├── capture.py              # Efficient screen capture
│   └── encoder.py              # Frame compressor/encoder
│
├── client/
│   ├── receiver_udp.py         # Python UDP client GUI
│   ├── receiver_tcp.py         # TCP version
│   ├── display_tkinter.py      # Tkinter GUI display
│   └── display_opencv.py       # OpenCV fallback viewer
│
├── utils/
│   └── config.py               # Buffer sizes, host settings
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Tech Stack

| Layer            | Tool/Library                |
|------------------|----------------------------|
| Screen Capture   | mss, pyvirtualdisplay, PIL |
| Image Encoding   | Pillow, io, zlib or lz4    |
| Network          | socket, threading          |
| GUI Display      | tkinter, OpenCV            |
| CLI/Utils        | argparse, threading, time  |

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/multi-lan-screen-share.git
   cd multi-lan-screen-share
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - `tkinter` is included with Python standard library.
   - `pyvirtualdisplay` is only needed for headless Linux/Raspberry Pi.

---

## Usage

### 1. Start the Broadcaster (Server)

#### **UDP Mode** (fast, best for LAN)
```bash
python -m server.broadcaster_udp
```

#### **TCP Mode** (reliable, slightly higher latency)
```bash
python -m server.broadcaster_tcp
```

---

### 2. Start a Viewer (Client)

#### **UDP Client (Tkinter GUI)**
```bash
python -m client.receiver_udp --host 0.0.0.0 --display tk
```

#### **UDP Client (OpenCV GUI)**
```bash
python -m client.receiver_udp --host 0.0.0.0 --display cv
```

#### **TCP Client (Tkinter GUI)**
```bash
python -m client.receiver_tcp --host <server_ip> --display tk
```

#### **TCP Client (OpenCV GUI)**
```bash
python -m client.receiver_tcp --host <server_ip> --display cv
```

> Replace `<server_ip>` with the IP address of the broadcasting machine.

---

## Configuration

All tunable parameters are in `utils/config.py`:

```python
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT_UDP = 5001
DEFAULT_PORT_TCP = 5002
UDP_BUFFER_SIZE = 65507
TCP_BUFFER_SIZE = 1024 * 1024
COMPRESSION = 'zlib'  # Options: 'zlib', 'lz4', 'none'
JPEG_QUALITY = 70
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 20
```

- **COMPRESSION**: Use `'zlib'` (default), `'lz4'` (faster, needs lz4), or `'none'`.
- **JPEG_QUALITY**: Lower for faster/lower bandwidth, higher for better image.
- **FRAME_WIDTH/HEIGHT**: Set to match your display or desired resolution.
- **FRAME_RATE**: Lower for less CPU/network usage.

---

## Testing

A test script is provided to verify core functionality:

```bash
python test_screen_share.py
```

- Tests screen capture, encoding, UDP and TCP networking.
- GUI display tests are skipped by default for headless/CI compatibility.

---

## Use Cases

-  Mirror your screen to multiple laptops in a classroom.
-  Display Raspberry Pi GUI remotely.
-  Debug/test UI on another device over LAN.
-  Use in air-gapped environments with no internet.

---

##  Advanced

- **Headless Mode**: On Linux/Raspberry Pi, install `pyvirtualdisplay` for Xvfb-based virtual display.
- **Custom Compression**: Add new methods in `server/encoder.py`.
- **Multiple Clients**: TCP server supports many clients; UDP is broadcast.
- **Performance**: For best results, use wired LAN and adjust JPEG quality/frame size.

---

##  Roadmap / Future Features

-  Audio stream over LAN
-  Mouse/keyboard remote control
-  mDNS/Bonjour-based auto-discovery
-  Python GUI viewer for Android (Kivy)
-  Compression optimizations with `lz4`, `blosc`

---

## FAQ

- **Q: Why do I see `[WARN] Frame too large for UDP`?**
  - A: Lower the JPEG quality or frame size in `utils/config.py`.

- **Q: Can I use this over Wi-Fi?**
  - A: Yes, but wired LAN is recommended for best performance.

- **Q: How do I run on Raspberry Pi or headless server?**
  - A: Install `pyvirtualdisplay` and ensure Xvfb is available.

---

## Contributing

Pull requests and issues are welcome! Please see the guidelines in the repository.

---

## Author

Shivansh Katiyar
Github : @SK8-infi

---

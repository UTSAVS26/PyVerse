# SecureSock: Encrypted File Transfer over WebSockets

SecureSock is a secure, resumable, encrypted file transfer system using WebSockets in Python. It features end-to-end AES-256 encryption, resumable chunked transfer, file compression, SHA-256 integrity verification, JWT authentication, and a PyQt5 GUI.

---

## Features

- AES-256-CBC encryption with PKCS7 padding
- JWT-based authentication
- ZIP compression before transmission
- Resumable chunked transfers (1MB chunks)
- Real-time progress using `tqdm`
- SHA-256 file hash verification
- GUI using PyQt5 with password field

---

## Requirements

```bash
pip install websockets cryptography tqdm pyqt5 pyjwt
```

---

## Usage

### Start Server

```bash
python secure_file_transfer.py server
```

### Client CLI

```bash
python secure_file_transfer.py client path/to/your/file
```

### GUI

```bash
python secure_file_transfer.py
```

---

## Author

**Shivansh Katiyar**
🐙 [@SK8-infi](https://github.com/SK8-infi)

---

# 🔒 One-Time-View File Server

A secure, temporary file-sharing system that enables the transfer of sensitive files via a single-use link. Once the file is accessed (downloaded/viewed), or after a user-defined expiry time, the file is automatically deleted from the server to ensure ephemeral and private transmission.

---

## Project Overview

This system is ideal for sending confidential files (e.g., documents, credentials, licenses) over LAN or the Internet. It can optionally generate QR codes to make sharing across devices seamless.

---

## Key Features

- **Self-Destructing File Links**
  - Files are deleted after one successful download (default), or after a configurable timeout.
- **One-Time Access Token**
  - Each upload generates a secure, random URL token.
  - Links are cryptographically hard to guess.
- **Works Over LAN/Internet**
  - Zero configuration for LAN.
  - Optional ngrok or dynamic DNS integration for external access.
- **QR Code Support**
  - Auto-generate QR code for the file URL.
  - Useful for mobile-to-PC transfer without typing URLs.
- **Simple Web Interface + API**
  - Drag-and-drop file upload UI
  - JSON API for integration
- **Minimal Logs / No Tracking**
  - No user data stored.
  - Only access time + IP for single-use validation.

---

## Tech Stack

| Component         | Technology                |
|------------------|--------------------------|
| Backend          | Python (FastAPI)         |
| Frontend         | HTML + CSS               |
| Temporary Storage| Disk (uploads/)          |
| QR Code          | Python `qrcode` lib      |
| Security         | SHA256 tokens, HTTPS opt |
| Deployment       | LAN + ngrok/IPv6/reverse |

---

## Directory Structure

```
one-time-view-server/
├── server.py
├── requirements.txt
├── templates/
│   └── upload.html
├── static/
│   └── style.css
├── uploads/
│   └── <temporary files>
├── utils/
│   ├── token_generator.py
│   └── cleanup_scheduler.py
├── test_server.py
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd one-time-view-server
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn server:app --reload
```

### 4. Open in browser
Visit [http://localhost:8000](http://localhost:8000)

---

## Running Tests

This project uses `pytest` and `httpx` for testing.

```bash
python -m pytest test_server.py -s
```

All tests should pass, covering:
- File upload and one-time download
- Expiry logic
- QR code endpoint

---

## Configuration

- **Default Expiry:** 10 minutes (600 seconds)
- **Custom Expiry:** Set in the upload form (min 10s, max 3600s)
- **Uploads Directory:** All files and QR codes are stored in `uploads/` and deleted after access/expiry.

---

## Cleanup

A background thread periodically deletes expired or accessed files and their QR codes from the `uploads/` directory.

---

## Possible Enhancements
- View-only support (PDF preview instead of download)
- Password-protected links
- Zip multiple files into one bundle
- Expiry scheduling (e.g., available for 5 mins starting at 9PM)
- Optional message/note with file

---

## Use Cases
- Sharing confidential files temporarily (like resumes, ID cards)
- Private one-time sharing between devices (LAN/P2P)
- Developer sending builds or API keys securely
- Temporary license/activation key sharing

---

## Inspiration
Inspired by services like Firefox Send (discontinued), Snapdrop, and WeTransfer, this tool focuses on privacy, ephemeral access, and ease-of-use in LAN or small office setups.

---

## Author
**@SK8-infi**

---

## Goals
- Secure, one-time-use file sharing
- Self-deletion of files on access or timeout
- Intuitive drag-and-drop interface
- QR code generation
- Lightweight and portable deployment 
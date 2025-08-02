# 🔐 ClipCrypt: Encrypted Clipboard Manager

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Secure, searchable, and local-only clipboard history — all encrypted.**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)

## 🎯 Overview

ClipCrypt is a powerful, privacy-focused clipboard manager that securely stores your clipboard history using AES-GCM encryption. Unlike cloud-based clipboard managers, ClipCrypt operates entirely offline, ensuring your sensitive data never leaves your device.

### Why ClipCrypt?

- **🔒 Privacy First**: All data is encrypted and stored locally
- **⚡ Fast & Lightweight**: Minimal resource usage with efficient search
- **🛡️ Secure**: AES-GCM encryption with local key storage
- **🔍 Smart Search**: Fuzzy search across encrypted content
- **🏷️ Organized**: Tag and categorize your clipboard entries
- **🖥️ Cross-Platform**: Works on Windows, macOS, and Linux

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| **🔒 End-to-End Encryption** | Every clipboard entry is encrypted using AES-GCM with a locally stored key |
| **📋 Automatic Monitoring** | Continuously watches clipboard changes and stores new entries securely |
| **🔍 Fuzzy Search** | Search through encrypted content with partial text matching |
| **🏷️ Tagging System** | Organize entries with custom tags (e.g., "code", "emails", "commands") |
| **📊 Rich Metadata** | Store timestamps, source app, and entry size for each snippet |
| **🔄 Cross-Platform** | Consistent experience across Windows, macOS, and Linux |

### Advanced Features

| Feature | Description |
|---------|-------------|
| **⌨️ CLI Interface** | Full-featured command-line interface with colored output |
| **📈 Statistics** | View storage statistics and usage analytics |
| **🗑️ Smart Cleanup** | Delete individual entries or clear all data |
| **📋 Copy Back** | Copy any stored entry back to clipboard |
| **🔧 Configurable** | Custom configuration directory and monitoring intervals |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Method 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/shivanshkatiyar/clipcrypt.git
cd clipcrypt

# Install dependencies
pip install -r requirements.txt

# Install ClipCrypt
pip install -e .
```

### Method 2: Direct Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/shivanshkatiyar/clipcrypt.git
```

### Method 3: Manual Installation

```bash
# Download and extract the source
wget https://github.com/shivanshkatiyar/clipcrypt/archive/main.zip
unzip main.zip
cd clipcrypt-main

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

## 🎯 Quick Start

### 1. Start Clipboard Monitoring

```bash
# Start monitoring your clipboard
clipcrypt monitor
```

This will start watching your clipboard for changes. New entries are automatically encrypted and stored.

### 2. List Your Clipboard History

```bash
# View all clipboard entries
clipcrypt list

# View last 10 entries
clipcrypt list --limit 10
```

### 3. Search Your History

```bash
# Search for entries containing "password"
clipcrypt search password

# Search for entries containing "code"
clipcrypt search code
```

### 4. Retrieve Specific Entries

```bash
# Get entry with ID 1
clipcrypt get 1

# Copy entry 1 back to clipboard
clipcrypt copy 1
```

## 📖 Usage Guide

### Command Reference

#### Basic Commands

```bash
# Start clipboard monitoring
clipcrypt monitor

# List all entries
clipcrypt list [--limit N]

# Get specific entry
clipcrypt get <entry_id>

# Search entries
clipcrypt search <query>

# Show statistics
clipcrypt stats

# Show information
clipcrypt info
```

#### Management Commands

```bash
# Add tag to entry
clipcrypt tag <entry_id> <tag>

# Remove tag from entry
clipcrypt untag <entry_id> <tag>

# List entries by tag
clipcrypt bytag <tag>

# Delete entry
clipcrypt delete <entry_id>

# Copy entry to clipboard
clipcrypt copy <entry_id>

# Clear all entries
clipcrypt clear
```

#### Advanced Options

```bash
# Use custom configuration directory
clipcrypt --config-dir /path/to/config monitor

# Show version
clipcrypt --version

# Show help
clipcrypt --help
```

### Sample Workflow

```bash
# 1. Start monitoring
clipcrypt monitor

# 2. Copy some text (Ctrl+C)
# Entry is automatically stored and encrypted

# 3. List recent entries
clipcrypt list

# 4. Search for specific content
clipcrypt search "important"

# 5. Get full content of entry
clipcrypt get 1

# 6. Add tags for organization
clipcrypt tag 1 "work"
clipcrypt tag 1 "important"

# 7. Copy back to clipboard
clipcrypt copy 1
```
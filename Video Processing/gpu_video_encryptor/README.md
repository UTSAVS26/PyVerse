# GPU-Accelerated Video Encryptor

Encrypt and decrypt video files securely and quickly using AES + SHA-256 with optional GPU acceleration.

## Features

- AES CBC mode encryption
- SHA-256 password-based key derivation
- GPU acceleration using Numba or CuPy
- Supports large video files efficiently
- Optional integrity check via SHA-256

## Usage

### Encrypt:

```bash
python main.py encrypt input.mp4 yourpassword --gpu
```

### Decrypt:

```bash
python main.py decrypt output_encrypt.avi yourpassword --gpu
```

## Dependencies

See `requirements.txt`

## Author

**Shivansh Katiyar** SSOC
GitHub: @SK8-infi

# 🔐 Secure Image Encrypter/Decrypter using SHA-256 and AES in Python

This project is a Python-based CLI tool that securely encrypts and decrypts image files using a password-derived key via SHA-256 hashing and AES symmetric encryption. It ensures data confidentiality and optional integrity verification.

---

## 📦 Features

- 🔑 **Password-based encryption**: SHA-256 derived symmetric key.
- 🛡️ **AES encryption/decryption** (AES-256 CBC mode).
- 🔍 **Integrity verification**: Optional SHA-256 hash check.
- 💻 **Command-line interface**: Easy and interactive usage.
- 🧠 **Educational**: Learn about hashing + encryption in one project.
- 📁 **Any image format supported**: Works on raw bytes.

---

## 🛠️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/secure-image-encryption.git
cd secure-image-encryption
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install pycryptodomex
```

---

## 🚀 Usage

Run the tool using:

```bash
python image_crypto.py
```

### 🔐 Encryption

```
Choose mode (encrypt/decrypt): encrypt
Enter image path: secret.png
Enter password: myStrongPassword123
```

Result: Encrypted file saved as `secret.png.enc`.

### 🔓 Decryption

```
Choose mode (encrypt/decrypt): decrypt
Enter encrypted file path: secret.png.enc
Enter password: myStrongPassword123
```

Result: Decrypted image saved as `secret.png.dec.png`.

---

## 🧪 Example

```
Original Hash   : 3a1f0b...
Decrypted Hash  : 3a1f0b... (Match ✅)
```

---

## 📌 Notes

- The encrypted file includes the IV (initialization vector) prepended to the ciphertext.
- Passwords are not stored anywhere — use strong ones you can remember.
- Works for any file but optimized for images.

---

## 👨‍💻 Author

Made by @SK8-infi
import hashlib
import os
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad

# === Helper functions ===

def sha256_hash(password: str) -> bytes:
    """Hash the password using SHA-256 to create a symmetric key."""
    return hashlib.sha256(password.encode()).digest()

def get_file_hash(filepath: str) -> str:
    """Return SHA-256 hash of the given file."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (FileNotFoundError, PermissionError) as e:
        raise ValueError(f"Cannot read file {filepath}: {e}")

# === Encryption ===

def encrypt_image(image_path: str, password: str, output_path: str = None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")
    if not password.strip():
        raise ValueError("Password cannot be empty")

    key = sha256_hash(password)
    iv = get_random_bytes(16)

    try:
        with open(image_path, "rb") as f:
            plaintext = f.read()
    except (PermissionError, IOError) as e:
        raise ValueError(f"Cannot read input file: {e}")

    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

    encrypted_data = iv + ciphertext  # IV is prepended

    if not output_path:
        output_path = image_path + ".enc"

    try:
        with open(output_path, "wb") as f:
            f.write(encrypted_data)
    except (PermissionError, IOError) as e:
        raise ValueError(f"Cannot write output file: {e}")

    print(f"[+] Encrypted image saved to: {output_path}")
    print(f"[✓] Original image SHA-256 hash: {hashlib.sha256(plaintext).hexdigest()}")
# === Decryption ===

def decrypt_image(enc_path: str, password: str, output_path: str = None):
    key = sha256_hash(password)

    with open(enc_path, "rb") as f:
        enc_data = f.read()

    iv = enc_data[:16]
    ciphertext = enc_data[16:]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

    if not output_path:
        # Try to restore original filename if possible
        base = enc_path.replace(".enc", "")
        output_path = base + ".dec.png"

    with open(output_path, "wb") as f:
        f.write(plaintext)

    print(f"[+] Decrypted image saved to: {output_path}")
    print(f"[✓] Decrypted image SHA-256 hash: {hashlib.sha256(plaintext).hexdigest()}")

# === CLI Interface ===

def main():
    print("=== SHA-256 + AES Image Encrypter/Decrypter ===")
    mode = input("Choose mode (encrypt/decrypt): ").strip().lower()

    if mode == "encrypt":
        image_path = input("Enter image path: ").strip()
        password = input("Enter password: ").strip()
        encrypt_image(image_path, password)

    elif mode == "decrypt":
        enc_path = input("Enter encrypted file path: ").strip()
        password = input("Enter password: ").strip()
        decrypt_image(enc_path, password)

    else:
        print("Invalid mode. Please choose 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()

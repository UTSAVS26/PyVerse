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
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encrypted file not found: {enc_path}")
    if not password.strip():
        raise ValueError("Password cannot be empty")
        
    key = sha256_hash(password)

    try:
        with open(enc_path, "rb") as f:
            enc_data = f.read()
    except (PermissionError, IOError) as e:
        raise ValueError(f"Cannot read encrypted file: {e}")
    
    if len(enc_data) < 16:
        raise ValueError("Invalid encrypted file: too short to contain IV")

    iv = enc_data[:16]
    ciphertext = enc_data[16:]

    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    except ValueError as e:
        raise ValueError("Decryption failed. Check password or file integrity.") from e

    if not output_path:
        # Preserve original extension instead of hardcoding .png
        if enc_path.endswith(".enc"):
            base = enc_path[:-4]  # Remove .enc
            output_path = (
                base + ".dec" + os.path.splitext(base)[1]
                if os.path.splitext(base)[1]
                else base + ".dec"
            )
        else:
            output_path = enc_path + ".dec"

    try:
        with open(output_path, "wb") as f:
            f.write(plaintext)
    except (PermissionError, IOError) as e:
        raise ValueError(f"Cannot write decrypted file: {e}")

    print(f"[+] Decrypted image saved to: {output_path}")
    print(f"[✓] Decrypted image SHA-256 hash: {hashlib.sha256(plaintext).hexdigest()}")
# === CLI Interface ===

import getpass

def main():
    print("=== SHA-256 + AES Image Encrypter/Decrypter ===")

    try:
        mode = input("Choose mode (encrypt/decrypt): ").strip().lower()

        if mode == "encrypt":
            image_path = input("Enter image path: ").strip()
            if not image_path:
                print("Error: Image path cannot be empty")
                return
            password = getpass.getpass("Enter password: ")
            encrypt_image(image_path, password)

        elif mode == "decrypt":
            enc_path = input("Enter encrypted file path: ").strip()
            if not enc_path:
                print("Error: Encrypted file path cannot be empty")
                return
            password = getpass.getpass("Enter password: ")
            decrypt_image(enc_path, password)

        else:
            print("Invalid mode. Please choose 'encrypt' or 'decrypt'.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

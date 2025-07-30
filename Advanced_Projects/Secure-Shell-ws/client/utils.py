import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
from cryptography.hazmat.primitives import hashes

BLOCK_SIZE = 16
KEY_SIZE = 32  # AES-256


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt,
        iterations=100_000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    iv = os.urandom(BLOCK_SIZE)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext)


def decrypt(ciphertext_b64: bytes, key: bytes) -> bytes:
    data = base64.b64decode(ciphertext_b64)
    iv = data[:BLOCK_SIZE]
    ciphertext = data[BLOCK_SIZE:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize() 
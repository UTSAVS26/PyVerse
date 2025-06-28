
import hashlib
import os

def derive_key(password):
    if not password or not isinstance(password, str):
        raise ValueError("Password must be a non-empty string")
    return hashlib.sha256(password.encode()).digest()

def generate_iv():
    return os.urandom(16)

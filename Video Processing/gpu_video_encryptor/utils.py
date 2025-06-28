
import hashlib
import os

def derive_key(password):
    return hashlib.sha256(password.encode()).digest()

def generate_iv():
    return os.urandom(16)

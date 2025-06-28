
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

BLOCK_SIZE = 16

def encrypt_frame(frame_bytes, key, iv):
    if not frame_bytes:
        raise ValueError("Frame bytes cannot be empty")
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")

    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.encrypt(pad(frame_bytes, BLOCK_SIZE))

def decrypt_frame(enc_bytes, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc_bytes), BLOCK_SIZE)

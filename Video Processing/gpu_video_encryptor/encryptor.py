
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

BLOCK_SIZE = 16

def encrypt_frame(frame_bytes, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.encrypt(pad(frame_bytes, BLOCK_SIZE))

def decrypt_frame(enc_bytes, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc_bytes), BLOCK_SIZE)

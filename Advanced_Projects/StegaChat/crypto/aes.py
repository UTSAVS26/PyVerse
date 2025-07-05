from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Optional

def generate_key(password: str, salt: Optional[bytes] = None) -> tuple[Fernet, bytes]:
    """
    Generate a Fernet key from a password using PBKDF2.
    
    Args:
        password: The password to derive the key from
        salt: Optional salt. If None, a random salt will be generated
        
    Returns:
        Tuple of (Fernet object, salt)
    """
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return Fernet(key), salt

def encrypt_message(fernet: Fernet, message: str) -> bytes:
    """
    Encrypt a message using Fernet.
    
    Args:
        fernet: Fernet object for encryption
        message: Message to encrypt
        
    Returns:
        Encrypted message as bytes
    """
    try:
        return fernet.encrypt(message.encode())
    except Exception as e:
        raise ValueError(f"Encryption failed: {e}")

def decrypt_message(fernet: Fernet, token: bytes) -> str:
    """
    Decrypt a message using Fernet.
    
    Args:
        fernet: Fernet object for decryption
        token: Encrypted message as bytes
        
    Returns:
        Decrypted message as string
    """
    try:
        return fernet.decrypt(token).decode()
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")

def save_key_info(fernet: Fernet, salt: bytes, filepath: str):
    """Save key information (salt) to a file for later use."""
    try:
        with open(filepath, 'wb') as f:
            f.write(salt)
    except Exception as e:
        raise ValueError(f"Failed to save key info: {e}")

def load_key_info(filepath: str) -> bytes:
    """Load salt from a file."""
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to load key info: {e}")

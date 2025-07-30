"""
Encryption module for ClipCrypt.

Handles AES-GCM encryption and decryption of clipboard data
using a locally stored encryption key.
"""

import os
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend


class EncryptionManager:
    """Manages encryption and decryption of clipboard data."""
    
    def __init__(self, config_dir: Path):
        """Initialize the encryption manager.
        
        Args:
            config_dir: Directory to store encryption keys
        """
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = config_dir / "encryption.key"
        self._key: Optional[bytes] = None
        self._aesgcm: Optional[AESGCM] = None
        
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return AESGCM.generate_key(bit_length=256)
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate a new one."""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not load existing key: {e}")
        
        # Generate new key
        key = self._generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return key
    
    def _get_key(self) -> bytes:
        """Get the encryption key, loading it if necessary."""
        if self._key is None:
            self._key = self._load_or_generate_key()
        return self._key
    
    def _get_aesgcm(self) -> AESGCM:
        """Get the AESGCM instance."""
        if self._aesgcm is None:
            self._aesgcm = AESGCM(self._get_key())
        return self._aesgcm
    
    def encrypt(self, data: str) -> Dict[str, Any]:
        """Encrypt clipboard data.
        
        Args:
            data: The text data to encrypt
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        aesgcm = self._get_aesgcm()
        nonce = os.urandom(12)
        
        # Encrypt the data
        ciphertext = aesgcm.encrypt(nonce, data.encode('utf-8'), None)
        
        return {
            'encrypted_data': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'algorithm': 'AES-GCM'
        }
    
    def decrypt(self, encrypted_data: str, nonce: str) -> str:
        """Decrypt clipboard data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            nonce: Base64 encoded nonce
            
        Returns:
            Decrypted text data
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            aesgcm = self._get_aesgcm()
            ciphertext = base64.b64decode(encrypted_data)
            nonce_bytes = base64.b64decode(nonce)
            
            # Decrypt the data
            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext, None)
            return plaintext.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def is_encrypted(self, data: Dict[str, Any]) -> bool:
        """Check if data is encrypted.
        
        Args:
            data: Data dictionary to check
            
        Returns:
            True if data appears to be encrypted
        """
        return (
            isinstance(data, dict) and
            'encrypted_data' in data and
            'nonce' in data and
            'algorithm' in data
        )
    
    def change_key(self) -> bool:
        """Change the encryption key (requires re-encrypting all data).
        
        Returns:
            True if key was changed successfully
        """
        try:
            # Generate new key
            new_key = self._generate_key()
            
            # Backup old key
            if self.key_file.exists():
                backup_file = self.config_dir / "encryption.key.backup"
                with open(self.key_file, 'rb') as src, open(backup_file, 'wb') as dst:
                    dst.write(src.read())
            
            # Write new key
            with open(self.key_file, 'wb') as f:
                f.write(new_key)
            
            # Reset internal state
            self._key = new_key
            self._aesgcm = AESGCM(new_key)
            
            return True
        except Exception as e:
            print(f"Failed to change key: {e}")
            return False 
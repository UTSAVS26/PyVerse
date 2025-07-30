"""
Tests for the encryption module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from clipcrypt.encryption import EncryptionManager


class TestEncryptionManager:
    """Test cases for EncryptionManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def encryption_manager(self, temp_dir):
        """Create an EncryptionManager instance for testing."""
        return EncryptionManager(temp_dir)
    
    def test_initialization(self, encryption_manager, temp_dir):
        """Test EncryptionManager initialization."""
        assert encryption_manager.config_dir == temp_dir
        assert encryption_manager.key_file == temp_dir / "encryption.key"
        assert encryption_manager.key_file.exists()
    
    def test_key_generation(self, encryption_manager):
        """Test that encryption keys are generated correctly."""
        key = encryption_manager._generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 32  # AES-GCM 256-bit key
    
    def test_encrypt_decrypt(self, encryption_manager):
        """Test encryption and decryption of data."""
        test_data = "Hello, ClipCrypt! This is a test message."
        
        # Encrypt the data
        encrypted = encryption_manager.encrypt(test_data)
        
        # Check encrypted data structure
        assert 'encrypted_data' in encrypted
        assert 'nonce' in encrypted
        assert 'algorithm' in encrypted
        assert encrypted['algorithm'] == 'AES-GCM'
        
        # Decrypt the data
        decrypted = encryption_manager.decrypt(
            encrypted['encrypted_data'],
            encrypted['nonce']
        )
        
        # Verify decrypted data matches original
        assert decrypted == test_data
    
    def test_encrypt_decrypt_empty_string(self, encryption_manager):
        """Test encryption and decryption of empty string."""
        test_data = ""
        
        encrypted = encryption_manager.encrypt(test_data)
        decrypted = encryption_manager.decrypt(
            encrypted['encrypted_data'],
            encrypted['nonce']
        )
        
        assert decrypted == test_data
    
    def test_encrypt_decrypt_unicode(self, encryption_manager):
        """Test encryption and decryption of Unicode data."""
        test_data = "Hello, 世界! 🌍 This is a test with Unicode characters."
        
        encrypted = encryption_manager.encrypt(test_data)
        decrypted = encryption_manager.decrypt(
            encrypted['encrypted_data'],
            encrypted['nonce']
        )
        
        assert decrypted == test_data
    
    def test_encrypt_decrypt_large_data(self, encryption_manager):
        """Test encryption and decryption of large data."""
        test_data = "A" * 10000  # 10KB of data
        
        encrypted = encryption_manager.encrypt(test_data)
        decrypted = encryption_manager.decrypt(
            encrypted['encrypted_data'],
            encrypted['nonce']
        )
        
        assert decrypted == test_data
    
    def test_is_encrypted(self, encryption_manager):
        """Test the is_encrypted method."""
        # Test with encrypted data
        encrypted = encryption_manager.encrypt("test")
        assert encryption_manager.is_encrypted(encrypted) is True
        
        # Test with non-encrypted data
        assert encryption_manager.is_encrypted({"test": "data"}) is False
        assert encryption_manager.is_encrypted("plain text") is False
        assert encryption_manager.is_encrypted(None) is False
    
    def test_decrypt_invalid_data(self, encryption_manager):
        """Test decryption with invalid data."""
        with pytest.raises(ValueError):
            encryption_manager.decrypt("invalid_data", "invalid_nonce")
    
    def test_key_persistence(self, temp_dir):
        """Test that encryption keys are persisted correctly."""
        # Create first manager
        manager1 = EncryptionManager(temp_dir)
        key1 = manager1._get_key()
        
        # Create second manager (should load same key)
        manager2 = EncryptionManager(temp_dir)
        key2 = manager2._get_key()
        
        assert key1 == key2
    
    def test_change_key(self, encryption_manager):
        """Test changing the encryption key."""
        # Store some data with original key
        original_data = "test data"
        encrypted = encryption_manager.encrypt(original_data)
        
        # Change the key
        assert encryption_manager.change_key() is True
        
        # Try to decrypt with new key (should fail)
        with pytest.raises(ValueError):
            encryption_manager.decrypt(
                encrypted['encrypted_data'],
                encrypted['nonce']
            )
        
        # Encrypt new data with new key
        new_data = "new test data"
        new_encrypted = encryption_manager.encrypt(new_data)
        decrypted = encryption_manager.decrypt(
            new_encrypted['encrypted_data'],
            new_encrypted['nonce']
        )
        
        assert decrypted == new_data
    
    def test_multiple_encryptions(self, encryption_manager):
        """Test that multiple encryptions of the same data produce different results."""
        test_data = "test message"
        
        encrypted1 = encryption_manager.encrypt(test_data)
        encrypted2 = encryption_manager.encrypt(test_data)
        
        # Encrypted data should be different due to different nonces
        assert encrypted1['encrypted_data'] != encrypted2['encrypted_data']
        assert encrypted1['nonce'] != encrypted2['nonce']
        
        # But both should decrypt to the same original data
        decrypted1 = encryption_manager.decrypt(
            encrypted1['encrypted_data'],
            encrypted1['nonce']
        )
        decrypted2 = encryption_manager.decrypt(
            encrypted2['encrypted_data'],
            encrypted2['nonce']
        )
        
        assert decrypted1 == decrypted2 == test_data 
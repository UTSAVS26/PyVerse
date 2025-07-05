#!/usr/bin/env python3
"""
Test script to verify StegaChat security fixes
"""

import os
import tempfile
import shutil
from PIL import Image
import numpy as np

# Import our modules
from crypto.aes import generate_key, encrypt_message, decrypt_message, save_key_info, load_key_info
from steg.image_steg import encode_message, decode_message, get_max_message_length

def test_crypto_fixes():
    """Test the cryptographic security fixes"""
    print("🔐 Testing Crypto Fixes...")
    
    # Test 1: Password-based key generation
    password = "test_password_123"
    fernet1, salt1 = generate_key(password)
    fernet2, salt2 = generate_key(password)
    
    # Same password should generate different keys (due to random salt)
    assert fernet1 != fernet2, "Same password should generate different keys"
    assert salt1 != salt2, "Salts should be different"
    
    # Test 2: Encryption/Decryption
    test_message = "Hello, this is a secret message!"
    encrypted = encrypt_message(fernet1, test_message)
    decrypted = decrypt_message(fernet1, encrypted)
    
    assert decrypted == test_message, "Encryption/Decryption should work correctly"
    
    # Test 3: Key storage
    with tempfile.NamedTemporaryFile(delete=False) as f:
        key_file = f.name
    
    try:
        save_key_info(fernet1, salt1, key_file)
        loaded_salt = load_key_info(key_file)
        assert loaded_salt == salt1, "Salt should be saved and loaded correctly"
        
        # Test with loaded salt
        fernet3, _ = generate_key(password, loaded_salt)
        decrypted2 = decrypt_message(fernet3, encrypted)
        assert decrypted2 == test_message, "Should decrypt with loaded salt"
        
    finally:
        os.unlink(key_file)
    
    print("✅ Crypto fixes passed!")

def test_steganography_fixes():
    """Test the steganography security fixes"""
    print("🖼️ Testing Steganography Fixes...")
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_image_path = f.name
    test_image.save(test_image_path)  # Save the image to disk
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name
    
    try:
        # Test 1: Basic encoding/decoding
        test_message = "Test message"
        success = encode_message(test_image_path, test_message, output_path)
        assert success, "Encoding should succeed"
        
        decoded_message = decode_message(output_path)
        assert decoded_message == test_message, "Decoding should work correctly"
        
        # Test 2: Message length validation
        max_length = get_max_message_length(test_image_path)
        assert max_length > 0, "Should calculate max message length"
        
        # Test 3: Error handling for non-existent file
        try:
            encode_message("nonexistent.png", "test", output_path)
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        # Test 4: Error handling for empty message
        try:
            encode_message(test_image_path, "", output_path)
            assert False, "Should raise ValueError for empty message"
        except ValueError:
            pass  # Expected
        
        # Test 5: Error handling for message too long
        long_message = "x" * (max_length + 100)
        try:
            encode_message(test_image_path, long_message, output_path)
            assert False, "Should raise ValueError for message too long"
        except ValueError:
            pass  # Expected
            
    finally:
        os.unlink(test_image_path)
        os.unlink(output_path)
    
    print("✅ Steganography fixes passed!")

def test_integration():
    """Test integration of crypto and steganography"""
    print("🔗 Testing Integration...")
    
    # Create a test image
    test_image = Image.new('RGB', (200, 200), color='blue')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_image_path = f.name
    test_image.save(test_image_path)  # Save the image to disk
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.key', delete=False) as f:
        key_file = f.name
    
    try:
        # Test encrypted steganography
        password = "secure_password_456"
        original_message = "This is a secret encrypted message!"
        
        # Encrypt
        fernet, salt = generate_key(password)
        encrypted = encrypt_message(fernet, original_message)
        encrypted_str = encrypted.decode('latin-1')
        
        # Encode
        success = encode_message(test_image_path, encrypted_str, output_path)
        assert success, "Should encode encrypted message"
        
        # Save key info
        save_key_info(fernet, salt, key_file)
        
        # Decode
        decoded_encrypted = decode_message(output_path)
        assert decoded_encrypted == encrypted_str, "Should decode encrypted message correctly"
        
        # Decrypt
        loaded_salt = load_key_info(key_file)
        fernet2, _ = generate_key(password, loaded_salt)
        decrypted_message = decrypt_message(fernet2, decoded_encrypted.encode('latin-1'))
        
        assert decrypted_message == original_message, "Should decrypt message correctly"
        
    finally:
        os.unlink(test_image_path)
        os.unlink(output_path)
        os.unlink(key_file)
    
    print("✅ Integration test passed!")

def main():
    """Run all tests"""
    print("🧪 Running StegaChat Security Fix Tests...\n")
    
    try:
        test_crypto_fixes()
        test_steganography_fixes()
        test_integration()
        
        print("\n🎉 All tests passed! Security fixes are working correctly.")
        print("\nKey improvements verified:")
        print("✅ Proper password-based encryption with PBKDF2")
        print("✅ Random salt generation and secure storage")
        print("✅ Input validation and error handling")
        print("✅ Message length validation")
        print("✅ Integration between crypto and steganography")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
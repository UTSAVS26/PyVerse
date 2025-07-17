import unittest
import os
from client import utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.password = 'testpassword'
        self.salt = os.urandom(16)
        self.key = utils.derive_key(self.password, self.salt)
        self.data = b'Hello, Secure Shell!'

    def test_key_derivation(self):
        key2 = utils.derive_key(self.password, self.salt)
        self.assertEqual(self.key, key2)
        self.assertEqual(len(self.key), 32)

    def test_encrypt_decrypt(self):
        encrypted = utils.encrypt(self.data, self.key)
        decrypted = utils.decrypt(encrypted, self.key)
        self.assertEqual(decrypted, self.data)

if __name__ == '__main__':
    unittest.main() 
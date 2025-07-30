import unittest
import os
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server import auth

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.token = 'SECRET123'
        self.password = 'mypass'
        self.salt = os.urandom(16)

    def test_verify_token(self):
        self.assertTrue(auth.verify_token('SECRET123', self.token))
        self.assertFalse(auth.verify_token('WRONG', self.token))

    def test_derive_key(self):
        key1 = auth.derive_key(self.password, self.salt)
        key2 = auth.derive_key(self.password, self.salt)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 32)

if __name__ == '__main__':
    unittest.main() 
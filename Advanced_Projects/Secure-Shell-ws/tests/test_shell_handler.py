import unittest
from server import shell_handler

class TestShellHandler(unittest.TestCase):
    def test_execute_command(self):
        output = shell_handler.execute_command('echo Hello')
        self.assertIn('Hello', output)

    def test_invalid_command(self):
        output = shell_handler.execute_command('nonexistentcommand1234')
        self.assertTrue('Error' in output or output)

if __name__ == '__main__':
    unittest.main() 
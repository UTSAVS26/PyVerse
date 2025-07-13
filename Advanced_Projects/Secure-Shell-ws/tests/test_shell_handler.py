import unittest
from server import shell_handler

class TestShellHandler(unittest.TestCase):
    def test_execute_command(self):
        output = shell_handler.execute_command('echo Hello')
        self.assertIn('Hello', output)

    def test_invalid_command(self):
        output = shell_handler.execute_command('nonexistentcommand1234')
        # Should contain error indication or be a known error pattern
        self.assertTrue(
            'Error' in output
            or 'not found' in output
            or 'command not found' in output.lower()
        )

if __name__ == '__main__':
    unittest.main() 
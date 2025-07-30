import unittest
import os
from core.logger import Logger

class TestLogger(unittest.TestCase):
    def test_log_event_in_memory(self):
        logger = Logger()
        logger.log_event('file.py', 1, 'x = 1', {'x': 1}, 'reason')
        self.assertEqual(len(logger.logs), 1)
        self.assertEqual(logger.logs[0]['file'], 'file.py')

    def test_log_event_json(self):
        filename = 'test_log.json'
        logger = Logger(filename)
        logger.log_event('file.py', 2, 'y = 2', {'y': 2}, 'reason2')
        del logger  # triggers __del__
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            import json
            data = json.load(f)
            self.assertEqual(data[0]['line_no'], 2)
        os.remove(filename)

    def test_log_event_html(self):
         os.remove(filename)

    def test_log_event_plain_text(self):
        filename = 'test_log.txt'
        logger = Logger(filename)
        logger.log_event('file.py', 4, 'w = 4', {'w': 4}, 'reason4')
        del logger
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            content = f.read()
            self.assertIn('file.py:4', content)
            self.assertIn('w = 4', content)
            self.assertIn('reason4', content)
        os.remove(filename)

 if __name__ == '__main__':
        logger = Logger(filename)
        logger.log_event('file.py', 3, 'z = 3', {'z': 3}, 'reason3')
        del logger
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            content = f.read()
            self.assertIn('<html>', content)
            self.assertIn('file.py', content)
        os.remove(filename)

if __name__ == '__main__':
    unittest.main() 
import unittest
import os
from core.parser import QueryParser
from core.command_map import CommandMapper
from core.executor import CommandExecutor
from core.utils import get_system_path, log_command

class TestSmartCLI(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser()
        self.mapper = CommandMapper()
        self.executor = CommandExecutor()

    def test_parser_basic(self):
        q = "Delete all zip files in Downloads older than 1 month"
        parsed = self.parser.parse_query(q)
        self.assertIn(parsed['intent'], ['delete', 'find', 'compress', 'list'])
        self.assertIn('zip', str(parsed['entities']))
        self.assertIn('Downloads', str(parsed['entities']))

    def test_parser_various_queries(self):
        queries = [
            ("Find all PDFs from last week on Desktop", 'find', 'pdf', 'Desktop', 'last week'),
            ("Compress large images in Documents", 'compress', 'image', 'Documents', 'large'),
            ("List all files modified today in Downloads", 'list', 'file', 'Downloads', 'today'),
            ("Delete small jpg files in Desktop older than 2 weeks", 'delete', 'jpg', 'Desktop', 'older than 2 weeks'),
        ]
        for q, intent, ftype, loc, time in queries:
            parsed = self.parser.parse_query(q)
            self.assertIn(intent, str(parsed['intent']))
            self.assertIn(ftype, str(parsed['entities']))
            self.assertIn(loc, str(parsed['entities']))
            self.assertIn(time.split()[0], str(parsed['entities']))

    def test_command_mapping(self):
        test_cases = [
            ({'intent': 'delete', 'entities': {'file_type': ['zip'], 'location': ['Downloads'], 'time': ['older than 1 month']}, 'raw': ''}, 'find', 'Downloads', '-delete'),
            ({'intent': 'find', 'entities': {'file_type': ['pdf'], 'location': ['Desktop'], 'time': ['last week']}, 'raw': ''}, 'find', 'Desktop', '*.pdf'),
            ({'intent': 'compress', 'entities': {'file_type': ['image'], 'location': ['Documents']}, 'raw': ''}, 'zip', 'Documents', ''),
        ]
        for parsed, expect_cmd, expect_loc, expect_flag in test_cases:
            cmd = self.mapper.map_intent(parsed)
            self.assertIn(expect_cmd, cmd)
            self.assertIn(expect_loc, cmd)
            if expect_flag:
                self.assertIn(expect_flag, cmd)

    def test_executor_safety(self):
        dangerous = [
            "rm -rf /", "mkfs /dev/sda", "dd if=/dev/zero of=/dev/sda", "find /etc -name '*.conf' -delete",
            "shutdown now", "reboot", ":(){:|:&};:", "chmod 777 /"
        ]
        for cmd in dangerous:
            self.assertFalse(self.executor.is_safe(cmd), f"Should block: {cmd}")
        safe = [
            f"find {get_system_path('downloads')} -name '*.zip' -delete",
            f"find {get_system_path('desktop')} -name '*.pdf'",
            f"cd {get_system_path('documents')} && zip -r archive.zip *.pdf"
        ]
        for cmd in safe:
            self.assertTrue(self.executor.is_safe(cmd), f"Should allow: {cmd}")

    def test_utils_get_system_path(self):
        self.assertTrue(os.path.isdir(get_system_path('downloads')))
        self.assertTrue(os.path.isdir(get_system_path('desktop')))
        self.assertTrue(os.path.isdir(get_system_path('documents')))

    def test_log_command(self):
        log_command('echo test', True, logfile='test_smartcli.log')
        with open('test_smartcli.log') as f:
            content = f.read()
        self.assertIn('echo test', content)
        os.remove('test_smartcli.log')

    def test_integration_parser_to_command(self):
        queries = [
            ("Delete all zip files in Downloads older than 1 month", 'delete'),
            ("Find all PDFs from last week on Desktop", 'find'),
            ("Compress large images in Documents", 'compress'),
        ]
        for q, expected_intent in queries:
            parsed = self.parser.parse_query(q)
            cmd = self.mapper.map_intent(parsed)
            self.assertIsInstance(cmd, str)
            self.assertGreater(len(cmd), 10)
            if expected_intent == 'compress':
                self.assertIn('zip', cmd)
            else:
                self.assertIn('find', cmd)

    def test_edge_cases(self):
        # Unknown intent
        parsed = self.parser.parse_query("Blah blah nothing matches")
        self.assertIsNone(parsed['intent'])
        # No file type
        parsed = self.parser.parse_query("Delete in Downloads")
        self.assertIn('Downloads', str(parsed['entities']))
        # No location
        parsed = self.parser.parse_query("Delete all zip files older than 1 month")
        self.assertIn('zip', str(parsed['entities']))

if __name__ == '__main__':
    unittest.main() 
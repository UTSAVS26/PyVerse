import unittest
from core.ast_parser import ASTParser

class TestASTParser(unittest.TestCase):
    def setUp(self):
        self.source = (
            "x = 1\n"
            "if x > 0:\n"
            "    y = 2\n"
            "while y < 10:\n"
            "    y += 1\n"
            "def foo():\n"
            "    pass\n"
        )
        self.parser = ASTParser(self.source)

    def test_get_conditions(self):
        conditions = self.parser.get_conditions()
        self.assertTrue(any('x > 0' in cond for _, cond in conditions))
        self.assertTrue(any('y < 10' in cond for _, cond in conditions))

    def test_get_assignments(self):
        assignments = self.parser.get_assignments()
        self.assertTrue(any('x = 1' in assign for _, assign in assignments))
        self.assertTrue(any('y = 2' in assign or 'y += 1' in assign for _, assign in assignments))

    def test_get_control_flow(self):
        flow = self.parser.get_control_flow()
        types = [t for _, t in flow]
        self.assertIn('if', types)
        self.assertIn('while', types)
        self.assertIn('function foo', types)

if __name__ == '__main__':
    unittest.main() 
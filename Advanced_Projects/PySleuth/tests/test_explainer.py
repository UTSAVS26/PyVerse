import unittest
import types
from core.explainer import explain_branch

class DummyFrame:
    def __init__(self, f_globals, f_locals):
        self.f_globals = f_globals
        self.f_locals = f_locals

class TestExplainer(unittest.TestCase):
    def test_if_branch(self):
        frame = DummyFrame({}, {'x': 7})
        line = 'if x > 5:'
        result = explain_branch(frame, line)
        self.assertIn('Entered', result)
        self.assertIn('x > 5', result)
        self.assertIn('True', result)

    def test_while_branch(self):
        frame = DummyFrame({}, {'y': 3})
        line = 'while y < 10:'
        result = explain_branch(frame, line)
        self.assertIn('Entered', result)
        self.assertIn('y < 10', result)
        self.assertIn('True', result)

    def test_function_def(self):
        frame = DummyFrame({}, {'a': 1})
        line = 'def foo(a): pass'
        result = explain_branch(frame, line)
        self.assertIn('Entered function', result)
        self.assertIn('foo', result)

    def test_invalid_line(self):
        frame = DummyFrame({}, {})
        line = 'not valid python'
        result = explain_branch(frame, line)
        self.assertIn('Could not explain', result)

if __name__ == '__main__':
    unittest.main() 
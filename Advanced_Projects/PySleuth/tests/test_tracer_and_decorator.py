import unittest
import os
from core.tracer import Tracer
from decorators.trace import trace

class TestTracerAndDecorator(unittest.TestCase):
    def test_trace_function(self):
        filename = 'test_trace_log.json'
        tracer = Tracer(log_file=filename)
        def test_fn():
            x = 1
            if x == 1:
                y = 2
            return y
        # Simulate tracing
        import sys
        sys.settrace(tracer.trace_calls)
        result = test_fn()
        sys.settrace(None)
        self.assertEqual(result, 2)
        del tracer
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            import json
            data = json.load(f)
            self.assertTrue(any('Entered' in log['reason'] for log in data))
        os.remove(filename)

    def test_trace_decorator(self):
        filename = 'test_trace_decorator_log.json'
        tracer = Tracer(log_file=filename)
        @trace(tracer=tracer)
        def foo():
            a = 5
            if a > 2:
                b = 10
            return b
        result = foo()
        self.assertEqual(result, 10)
        del tracer
        self.assertTrue(os.path.exists(filename))
        with open(filename) as f:
            import json
            data = json.load(f)
            self.assertTrue(any('Entered' in log['reason'] for log in data))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main() 
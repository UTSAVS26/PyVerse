from functools import wraps
from core.tracer import Tracer

def trace(func=None, *, tracer=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            import sys
            used_tracer = tracer or Tracer()
            sys.settrace(used_tracer.trace_calls)
            result = f(*args, **kwargs)
            sys.settrace(None)
            return result
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator
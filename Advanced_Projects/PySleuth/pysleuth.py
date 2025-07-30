import argparse
from core.tracer import Tracer
from decorators.trace import trace

__version__ = '0.1.0'

def main():
    parser = argparse.ArgumentParser(description='PySleuth: Context-Aware Python Debugger')
    parser.add_argument('script', nargs='?', help='Python script to run')
    parser.add_argument('--tui', action='store_true', help='Enable TUI mode')
    parser.add_argument('--log', help='Log output file (json/txt/html)')
    parser.add_argument('--trace-fn', nargs='*', help='Specific function names to trace')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args()

    if not args.script:
        parser.print_help()
        exit(1)

    tracer = Tracer(tui=args.tui, log_file=args.log, target_fns=args.trace_fn)
    tracer.run_script(args.script)

if __name__ == '__main__':
    main()
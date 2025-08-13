import sys, runpy, linecache
from core.explainer import explain_branch
from core.logger import Logger

class Tracer:
    def __init__(self, tui=False, log_file=None, target_fns=None):
        self.logger = Logger(log_file)
        self.target_fns = target_fns or []
        self.tui = tui

    def trace_calls(self, frame, event, arg):
        if event == 'call' and (not self.target_fns or frame.f_code.co_name in self.target_fns):
            return self.trace_lines
        return

    def trace_lines(self, frame, event, arg):
        if event == 'line':
            lineno = frame.f_lineno
            filename = frame.f_globals.get('__file__', '')
            line = linecache.getline(filename, lineno).strip()
            branch_reason = explain_branch(frame, line)
            self.logger.log_event(filename, lineno, line, frame.f_locals, branch_reason)
        return self.trace_lines

    def run_script(self, script):
        if self.tui:
            try:
                from tui.curses_ui import CursesUI
                ui = CursesUI()
                ui.run()
            except ImportError:
                print("TUI mode is not available. Please install curses and ensure tui/curses_ui.py exists.")
                return
        sys.settrace(self.trace_calls)
        runpy.run_path(script, run_name='__main__')
        sys.settrace(None)
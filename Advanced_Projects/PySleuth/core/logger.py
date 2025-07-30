import json

class Logger:
    def __init__(self, filename=None):
        self.filename = filename
        self.logs = []

    def log_event(self, file, lineno, line, locals_, reason):
        # Convert frame locals to a regular dict if needed
        try:
            safe_locals = dict(locals_)
        except Exception:
            safe_locals = str(locals_)
        log = {
            'file': file,
            'line_no': lineno,
            'code': line,
            'locals': safe_locals,
            'reason': reason
        }
        self.logs.append(log)
        print(f"{file}:{lineno} => {line}\n  Locals: {safe_locals}\n  Reason: {reason}\n")

    def __del__(self):
        if self.filename:
            with open(self.filename, 'w', encoding='utf-8') as f:
                if self.filename.endswith('.json'):
                    json.dump(self.logs, f, indent=2)
                elif self.filename.endswith('.html'):
                    f.write('<html><body><h2>PySleuth Trace Log</h2><ul>')
                    for log in self.logs:
                        f.write(f"<li><b>{log['file']}:{log['line_no']}</b>: {log['code']}<br>Locals: {log['locals']}<br>Reason: {log['reason']}</li>")
                    f.write('</ul></body></html>')
                else:
                    for log in self.logs:
                        f.write(f"{log['file']}:{log['line_no']} | {log['code']} | Locals: {log['locals']} | Reason: {log['reason']}\n")
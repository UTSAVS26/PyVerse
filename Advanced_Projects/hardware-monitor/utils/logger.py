import csv
import time

class MetricsLogger:
    def __init__(self):
        self.logs = []

import threading

class MetricsLogger:
    def __init__(self):
        self.logs = []
        self._lock = threading.Lock()

    def log(self, metrics: dict):
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")
        
        with self._lock:
            entry = {'timestamp': time.time()}
            entry.update(metrics)
            self.logs.append(entry)

    def export_csv(self, path):
        if not self.logs:
            return
        keys = self.logs[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.logs)

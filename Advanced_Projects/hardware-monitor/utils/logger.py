import csv
import time

class MetricsLogger:
    def __init__(self):
        self.logs = []

    def log(self, metrics: dict):
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

import json
import os

class Config:
    def __init__(self, path=None):
        self.defaults = {
            'cpu_threshold': 90,
            'ram_threshold': 90,
            'gpu_threshold': 90,
            'update_interval': 1,  # seconds
            'prediction_interval': 60  # seconds
        }
        self.path = path
        self.config = self.defaults.copy()
        if path and os.path.exists(path):
            self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f)

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("No path specified for saving config.")
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key):
        return self.config.get(key, self.defaults.get(key))

    def set(self, key, value):
        self.config[key] = value

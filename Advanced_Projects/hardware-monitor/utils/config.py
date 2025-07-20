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
        try:
            with open(path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self.config = self.defaults.copy()
                self.config.update(loaded_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load config from {path}: {e}. Using defaults.")
            self.config = self.defaults.copy()

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("No path specified for saving config.")
        try:
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save config to {path}: {e}")
            return False

    def get(self, key):
        return self.config.get(key, self.defaults.get(key))

    def set(self, key, value):
        self.config[key] = value

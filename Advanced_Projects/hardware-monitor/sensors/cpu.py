import psutil
import time
import sys

# Get per-core and total CPU usage (%)
def get_cpu_usage():
    return {
        'per_core': psutil.cpu_percent(percpu=True),
        'total': psutil.cpu_percent()
    }

# Get CPU temperature (if available)
def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                if 'cpu' in entry.label.lower() or 'core' in entry.label.lower():
                    return entry.current
        return None
    except (AttributeError, NotImplementedError):
        return None

# Collect CPU usage history for the last N seconds (1 value/sec)
def get_cpu_usage_history(seconds=60):
    history = []
    for _ in range(seconds):
        usage = psutil.cpu_percent()
        history.append(usage)
        time.sleep(1)
    return history

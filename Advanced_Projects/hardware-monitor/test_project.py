import matplotlib
matplotlib.use('Agg')
import pytest
import numpy as np

# Sensors
from sensors import cpu, gpu, memory
# Models
from models.lstm_predictor import LSTMPredictor
from models.stats_fallback import StatsFallbackPredictor
# Utils
from utils.logger import MetricsLogger
from utils.config import Config
# Visualizations
import visualizations.live_plot as live_plot
import visualizations.prediction_overlay as prediction_overlay

def test_cpu_sensor():
    usage = cpu.get_cpu_usage()
    assert 'per_core' in usage and 'total' in usage
    assert isinstance(usage['per_core'], list)
    assert isinstance(usage['total'], float)
    # Temperature may be None
    temp = cpu.get_cpu_temperature()
    assert temp is None or isinstance(temp, (float, int))

def test_memory_sensor():
    stats = memory.get_memory_stats()
    assert 'ram_total' in stats and 'ram_used' in stats
    assert stats['ram_total'] >= stats['ram_used']

def test_gpu_sensor():
    stats = gpu.get_gpu_stats()
    assert isinstance(stats, dict)
    # Accept error if no GPU
    if 'error' not in stats:
        assert 'gpu_util' in stats

def test_lstm_predictor():
    data = np.random.rand(120)
    model = LSTMPredictor(window_size=10, pred_steps=5)
    model.fit(data)
    preds = model.predict(data)
    assert len(preds) == 5

def test_stats_fallback_predictor():
    data = np.random.rand(120)
    model = StatsFallbackPredictor(order=(1,1,0), pred_steps=5)
    model.fit(data)
    preds = model.predict(data)
    assert len(preds) == 5

def test_logger():
    logger = MetricsLogger()
    logger.log({'cpu': 10, 'ram': 20})
    assert len(logger.logs) == 1
    logger.export_csv('test_log.csv')
    import os
    assert os.path.exists('test_log.csv')
    os.remove('test_log.csv')

def test_config():
    config = Config()
    assert config.get('cpu_threshold') == 90
    config.set('cpu_threshold', 80)
    assert config.get('cpu_threshold') == 80
    config.save('test_config.json')
    config2 = Config('test_config.json')
    assert config2.get('cpu_threshold') == 80
    import os
    os.remove('test_config.json')

def test_live_plot(monkeypatch):
    # Patch plt.show to avoid opening a window
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    data = np.random.rand(10)
    live_plot.live_plot(data, label="Test", export_path="test_plot.png")
    import os
    assert os.path.exists('test_plot.png')
    os.remove('test_plot.png')

def test_prediction_overlay(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    actual = np.random.rand(10)
    predicted = np.random.rand(5)
    prediction_overlay.overlay_prediction(actual, predicted, label="Test", export_path="test_overlay.png")
    import os
    assert os.path.exists('test_overlay.png')
    os.remove('test_overlay.png') 
# GPU stats via pynvml

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
except Exception:
    NVML_AVAILABLE = False

def get_gpu_stats():
    if not NVML_AVAILABLE:
        return {'error': 'pynvml not available or no NVIDIA GPU detected'}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            'gpu_util': util.gpu,  # percent
            'mem_util': util.memory,  # percent
            'vram_total': mem.total,
            'vram_used': mem.used,
            'vram_free': mem.free,
            'temperature': temp
        }
    except Exception as e:
        return {'error': str(e)}

import psutil

def get_memory_stats():
    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'ram_total': vm.total,
            'ram_used': vm.used,
            'ram_free': vm.available,
            'ram_percent': vm.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_free': swap.free,
            'swap_percent': swap.percent
        }
    except Exception as e:
        return {'error': f'Failed to retrieve memory stats: {str(e)}'}

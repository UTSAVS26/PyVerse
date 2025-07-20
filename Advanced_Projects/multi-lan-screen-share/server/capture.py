
import mss
from PIL import Image
try:
    from pyvirtualdisplay import Display
    _has_virtual = True
except ImportError:
    _has_virtual = False

def start_virtual_display(width=1280, height=720):
    if _has_virtual:
        display = Display(visible=0, size=(width, height))
        display.start()
        return display
    return None

def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if region is None else region
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img

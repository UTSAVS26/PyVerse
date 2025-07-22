
import socket
import time
from . import capture, encoder
from utils import config

def main():
def main():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        target = ('<broadcast>', config.DEFAULT_PORT_UDP)
        print(f"[UDP] Broadcasting on {target}")
    except OSError as e:
        print(f"[ERROR] Failed to create UDP socket: {e}")
        return

    # … rest of your broadcasting loop …
    
    # Optionally start virtual display for headless
    display = capture.start_virtual_display(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    try:
        while True:
            try:
                img = capture.capture_screen()
                frame = encoder.encode_jpeg(img)
                data = encoder.compress_data(frame)
            except Exception as e:
                print(f"[ERROR] Failed to capture/encode frame: {e}")
                time.sleep(1 / config.FRAME_RATE)
                continue

            # UDP max size
            if len(data) > config.UDP_BUFFER_SIZE:
                print(f"[WARN] Frame too large for UDP: {len(data)} bytes")
                continue
            sock.sendto(data, target)
            time.sleep(1 / config.FRAME_RATE)
    except KeyboardInterrupt:
        print("[UDP] Broadcast stopped.")
    finally:
        if display:
            display.stop()
        sock.close()

if __name__ == '__main__':
    main()

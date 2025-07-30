
import socket
import threading
import time
from . import capture, encoder
from utils import config

def handle_client(conn, addr, get_frame):
    print(f"[TCP] Client connected: {addr}")
    try:
        while True:
            frame = get_frame()
            size = len(frame).to_bytes(4, 'big')
            conn.sendall(size + frame)
    except Exception as e:
        print(f"[TCP] Client {addr} disconnected: {e}")
    finally:
        conn.close()

def main():
def main():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((config.DEFAULT_HOST, config.DEFAULT_PORT_TCP))
        sock.listen(5)
        print(f"[TCP] Broadcasting on {config.DEFAULT_HOST}:{config.DEFAULT_PORT_TCP}")
    except OSError as e:
        print(f"[TCP] Failed to start server: {e}")
        return
    
    display = capture.start_virtual_display(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    latest_frame = b''
    lock = threading.Lock()

    def get_frame():
        with lock:
            return latest_frame

    def capture_loop():
        nonlocal latest_frame
        while True:
            try:
                img = capture.capture_screen()
                frame = encoder.encode_jpeg(img)
                data = encoder.compress_data(frame)
                with lock:
                    latest_frame = data
            except Exception as e:
                print(f"[TCP] Capture error: {e}")
            time.sleep(1 / config.FRAME_RATE)

    threading.Thread(target=capture_loop, daemon=True).start()

    try:
        while True:
            conn, addr = sock.accept()
            threading.Thread(target=handle_client, args=(conn, addr, get_frame), daemon=True).start()
    except KeyboardInterrupt:
        print("[TCP] Broadcast stopped.")
    finally:
        if display:
            display.stop()
        sock.close()

if __name__ == '__main__':
    main()

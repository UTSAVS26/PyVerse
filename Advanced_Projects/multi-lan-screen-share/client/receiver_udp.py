
import socket
import argparse
from utils import config
from server import encoder
import threading
import time
import importlib

def receive_frames(host, display_callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, config.DEFAULT_PORT_UDP))
    print(f"[UDP] Listening on {host}:{config.DEFAULT_PORT_UDP}")
    while True:
        data, _ = sock.recvfrom(config.UDP_BUFFER_SIZE)
        try:
            frame = encoder.decompress_data(data)
            display_callback(frame)
        except Exception as e:
            print(f"[UDP] Frame error: {e}")

def main():
    parser = argparse.ArgumentParser(description='UDP Screen Receiver')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--display', type=str, default='tk', choices=['tk', 'cv'], help='Display method: tk or cv')
    args = parser.parse_args()

    if args.display == 'tk':
        display_loop = importlib.import_module('client.display_tkinter').display_loop
    else:
        display_loop = importlib.import_module('client.display_opencv').display_loop

from queue import Queue

    if args.display == 'tk':
        display_loop = importlib.import_module('client.display_tkinter').display_loop
    else:
        display_loop = importlib.import_module('client.display_opencv').display_loop

    frame_queue = Queue()
    def display_callback(frame):
        frame_queue.put(frame)

    threading.Thread(target=receive_frames, args=(args.host, display_callback), daemon=True).start()
    display_loop(frame_queue)

if __name__ == '__main__':
    main()

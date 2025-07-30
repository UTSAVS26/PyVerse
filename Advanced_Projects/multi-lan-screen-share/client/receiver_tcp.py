
import socket
import argparse
from utils import config
from server import encoder
import threading
import time
import importlib

def receive_frames(host, display_callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, config.DEFAULT_PORT_TCP))
    print(f"[TCP] Connected to {host}:{config.DEFAULT_PORT_TCP}")
    while True:
        size = sock.recv(4)
        if not size:
            break
        frame_size = int.from_bytes(size, 'big')
        data = b''
        while len(data) < frame_size:
            packet = sock.recv(min(config.TCP_BUFFER_SIZE, frame_size - len(data)))
            if not packet:
                break
            data += packet
        try:
            frame = encoder.decompress_data(data)
            display_callback(frame)
        except Exception as e:
            print(f"[TCP] Frame error: {e}")

def main():
    parser = argparse.ArgumentParser(description='TCP Screen Receiver')
    parser.add_argument('--host', type=str, required=True, help='Server host to connect to')
    parser.add_argument('--display', type=str, default='tk', choices=['tk', 'cv'], help='Display method: tk or cv')
    args = parser.parse_args()

    if args.display == 'tk':
        display_loop = importlib.import_module('client.display_tkinter').display_loop
    else:
        display_loop = importlib.import_module('client.display_opencv').display_loop

from queue import Queue

def main():
    parser = argparse.ArgumentParser(description='TCP Screen Receiver')
    parser.add_argument('--host', type=str, required=True, help='Server host to connect to')
    parser.add_argument('--display', type=str, default='tk', choices=['tk', 'cv'], help='Display method: tk or cv')
    args = parser.parse_args()

    if args.display == 'tk':
        display_loop = importlib.import_module('client.display_tkinter').display_loop
    else:
        display_loop = importlib.import_module('client.display_opencv').display_loop

    frame_queue = Queue()
    def display_callback(frame):
        frame_queue.put(frame)

    threading.Thread(
        target=receive_frames,
        args=(args.host, display_callback),
        daemon=True
    ).start()
    display_loop(frame_queue)

if __name__ == '__main__':
    main()

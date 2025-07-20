import subprocess
import sys
import time
import os

# Smoke test for capture and encoder
print('Testing capture and encoder modules...')
try:
    from server import capture, encoder
    img = capture.capture_screen()
    assert img is not None, 'Capture failed'
    jpeg = encoder.encode_jpeg(img)
    assert isinstance(jpeg, bytes) and len(jpeg) > 0, 'JPEG encoding failed'
    compressed = encoder.compress_data(jpeg)
    decompressed = encoder.decompress_data(compressed)
    assert decompressed == jpeg, 'Compression roundtrip failed'
    print('Capture and encoder tests passed.')
except Exception as e:
    print(f'Capture/encoder test failed: {e}')
    sys.exit(1)

# --- GUI display tests skipped ---
# (all GUI test code is commented out)

# Integration test: run UDP broadcaster and UDP client
print('Testing UDP broadcaster and receiver...')
server_proc = None
client_proc = None
try:
    # Start UDP broadcaster
    server_proc = subprocess.Popen([sys.executable, '-m', 'server.broadcaster_udp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Give server time to start
    # Start UDP client (OpenCV, will open a window)
    client_proc = subprocess.Popen([sys.executable, '-m', 'client.receiver_udp', '--host', '0.0.0.0', '--display', 'cv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Let them run for a few seconds
    # Check if server and client are still running
    if server_proc.poll() is not None:
        out, err = server_proc.communicate()
        print('Server exited early!')
        print(out.decode(), err.decode())
        raise RuntimeError('UDP broadcaster failed')
    if client_proc.poll() is not None:
        out, err = client_proc.communicate()
        print('Client exited early!')
        print(out.decode(), err.decode())
        raise RuntimeError('UDP receiver failed')
    print('UDP broadcaster and receiver appear to be running.')
finally:
    if client_proc:
        client_proc.terminate()
        client_proc.wait()
    if server_proc:
        server_proc.terminate()
        server_proc.wait()
    print('UDP test processes terminated.')

# Integration test: run TCP broadcaster and TCP client
print('Testing TCP broadcaster and receiver...')
server_proc = None
client_proc = None
try:
    # Start TCP broadcaster
    server_proc = subprocess.Popen([sys.executable, '-m', 'server.broadcaster_tcp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Give server time to start
    # Start TCP client (OpenCV, will open a window)
    client_proc = subprocess.Popen([sys.executable, '-m', 'client.receiver_tcp', '--host', '127.0.0.1', '--display', 'cv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Let them run for a few seconds
    # Check if server and client are still running
    if server_proc.poll() is not None:
        out, err = server_proc.communicate()
        print('Server exited early!')
        print(out.decode(), err.decode())
        raise RuntimeError('TCP broadcaster failed')
    if client_proc.poll() is not None:
        out, err = client_proc.communicate()
        print('Client exited early!')
        print(out.decode(), err.decode())
        raise RuntimeError('TCP receiver failed')
    print('TCP broadcaster and receiver appear to be running.')
finally:
    if client_proc:
        client_proc.terminate()
        client_proc.wait()
    if server_proc:
        server_proc.terminate()
        server_proc.wait()
    print('TCP test processes terminated.')

print('All tests completed.') 
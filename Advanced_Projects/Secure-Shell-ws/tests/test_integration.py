import subprocess
import sys
import time
import os
import signal

SERVER_CMD = [
    sys.executable, 'server/server.py',
    '--port', '8765',
    '--token', 'SECRET123',
    '--password', 'mypass'
]
CLIENT_CMD = [
    sys.executable, 'client/client.py',
    '--host', '127.0.0.1',
    '--port', '8765',
    '--token', 'SECRET123',
    '--password', 'mypass',
    '--command', 'echo integration_test'
]

def test_integration():
    # Start server
    server_proc = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Give server time to start
    try:
        # Start client and capture output
        client_proc = subprocess.Popen(CLIENT_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = client_proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            client_proc.kill()
            stdout, stderr = client_proc.communicate()
        output = stdout.decode(errors='ignore') + stderr.decode(errors='ignore')
        assert 'integration_test' in output
    finally:
        if os.name == 'nt':
            server_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            server_proc.terminate()
        server_proc.wait(timeout=5)

if __name__ == '__main__':
    test_integration()
    print('Integration test passed!') 
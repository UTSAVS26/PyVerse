import subprocess
import sys
import time
import os
import signal

import os

# Get credentials from environment or use test defaults
TEST_TOKEN = os.getenv('TEST_TOKEN', 'test_token_' + os.urandom(8).hex())
TEST_PASSWORD = os.getenv('TEST_PASSWORD', 'test_pass_' + os.urandom(8).hex())

SERVER_CMD = [
    sys.executable, 'server/server.py',
    '--port', '8765',
    '--token', TEST_TOKEN,
    '--password', TEST_PASSWORD
]
CLIENT_CMD = [
    sys.executable, 'client/client.py',
    '--host', '127.0.0.1',
    '--port', '8765',
    '--token', TEST_TOKEN,
    '--password', TEST_PASSWORD,
    '--command', 'echo integration_test'
]

import psutil

def test_integration():
    server_proc = None
    client_proc = None
    # Start server
    try:
        server_proc = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to be ready (more reliable than fixed sleep)
        import socket
        for _ in range(30):  # 30 second timeout
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 8765))
                sock.close()
                if result == 0:
                    break
            except:
                pass
            time.sleep(0.1)
        else:
            raise Exception("Server failed to start within timeout")
        
        # Start client and capture output
        client_proc = subprocess.Popen(CLIENT_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = client_proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            client_proc.kill()
            stdout, stderr = client_proc.communicate()
            raise Exception("Client timed out")
        
        if client_proc.returncode != 0:
            raise Exception(f"Client failed with exit code {client_proc.returncode}: {stderr.decode()}")
            
        output = stdout.decode(errors='ignore') + stderr.decode(errors='ignore')
        assert 'integration_test' in output
    finally:
        # Clean up processes more reliably
        for proc in [client_proc, server_proc]:
            if proc and proc.poll() is None:
                try:
                    if os.name == 'nt':
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                    proc.wait(timeout=5)
                except:
                    proc.kill()
                    proc.wait()
if __name__ == '__main__':
    test_integration()
    print('Integration test passed!') 
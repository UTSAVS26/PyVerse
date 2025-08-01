#!/usr/bin/env python3
"""
Test malicious script for behavior-based script detector.

This script contains various suspicious patterns to test the detector's
ability to identify potentially harmful behavior.
"""

import os
import subprocess
import base64
import pickle
import marshal
import socket
import urllib.request
import requests
import threading
import multiprocessing
import ctypes
import winreg
import sys

# Dangerous function calls
def dangerous_exec():
    """Contains exec usage."""
    code = "print('Hello from exec')"
    exec(code)  # Line 25: exec usage

def dangerous_eval():
    """Contains eval usage."""
    expression = "2 + 2"
    result = eval(expression)  # Line 30: eval usage
    return result

def shell_commands():
    """Contains subprocess and os.system calls."""
    # Line 35: subprocess usage
    subprocess.run(['ls', '-la'], capture_output=True)
    
    # Line 37: os.system usage
    os.system('echo "Hello from system"')
    
    # Line 39: os.popen usage
    output = os.popen('whoami').read()

def unsafe_deserialization():
    """Contains pickle and marshal usage."""
    # Line 44: pickle usage
    data = pickle.dumps({'key': 'value'})
    loaded = pickle.loads(data)
    
    # Line 47: marshal usage
    marshaled = marshal.dumps([1, 2, 3])
    unmarshaled = marshal.loads(marshaled)

def sensitive_file_access():
    """Contains access to sensitive file paths."""
    # Line 52: sensitive file access
    with open('/etc/passwd', 'r') as f:
        passwd_content = f.read()
    
    # Line 55: sensitive file access
    ssh_key_path = '~/.ssh/id_rsa'
    
    # Line 57: Windows sensitive paths
    windows_path = 'C:\\Windows\\System32\\config\\SAM'

def file_deletion():
    """Contains file deletion operations."""
    # Line 62: file deletion
    os.remove('temp_file.txt')
    
    # Line 64: file deletion
    os.unlink('another_temp.txt')
    
    # Line 66: directory deletion
    import shutil
    shutil.rmtree('temp_directory')

def network_operations():
    """Contains network download operations."""
    # Line 71: network download
    urllib.request.urlretrieve('http://example.com/file.txt', 'downloaded_file.txt')
    
    # Line 73: requests download
    response = requests.get('https://api.example.com/data')
    
    # Line 75: socket operations
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 8080))

def encoding_operations():
    """Contains encoding/decoding operations."""
    # Line 80: base64 encoding
    encoded = base64.b64encode(b'secret data')
    decoded = base64.b64decode(encoded)
    
    # Line 83: zlib compression
    import zlib
    compressed = zlib.compress(b'data to compress')
    decompressed = zlib.decompress(compressed)

def process_creation():
    """Contains process and thread creation."""
    # Line 89: thread creation
    thread = threading.Thread(target=lambda: print('Thread running'))
    thread.start()
    
    # Line 92: process creation
    process = multiprocessing.Process(target=lambda: print('Process running'))
    process.start()
    
    # Line 95: os.fork (Unix only)
    try:
        pid = os.fork()
    except AttributeError:
        pass  # Windows doesn't have fork

def suspicious_imports():
    """Contains suspicious module imports."""
    # These imports are already at the top of the file
    # Line 4-15: suspicious imports
    pass

def environment_manipulation():
    """Contains environment variable manipulation."""
    # Line 105: environment manipulation
    os.environ['CUSTOM_VAR'] = 'custom_value'
    
    # Line 107: environment manipulation
    os.putenv('ANOTHER_VAR', 'another_value')

def registry_access():
    """Contains Windows registry access."""
    # Line 112: registry access
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion')
    except (ImportError, OSError):
        pass  # Not on Windows or no access

def obfuscated_code():
    """Contains potentially obfuscated code."""
    # Line 119: obfuscated code
    obfuscated_string = "\\x48\\x65\\x6c\\x6c\\x6f"  # "Hello" in hex
    
    # Line 121: Unicode escape sequences
    unicode_string = "\\u0048\\u0065\\u006c\\u006c\\u006f"  # "Hello" in Unicode

def main():
    """Main function that calls all suspicious functions."""
    print("Testing suspicious behavior patterns...")
    
    # Call all the suspicious functions
    dangerous_exec()
    dangerous_eval()
    shell_commands()
    unsafe_deserialization()
    sensitive_file_access()
    file_deletion()
    network_operations()
    encoding_operations()
    process_creation()
    suspicious_imports()
    environment_manipulation()
    registry_access()
    obfuscated_code()
    
    print("All suspicious patterns tested!")

if __name__ == "__main__":
    main() 
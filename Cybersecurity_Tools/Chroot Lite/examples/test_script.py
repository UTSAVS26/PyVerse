#!/usr/bin/env python3
"""
Test script for Chroot Lite sandbox.
This script demonstrates various operations that can be performed in the sandbox.
"""

import os
import sys
import time
import math
import platform

def test_basic_operations():
    """Test basic Python operations."""
    print("=== Basic Operations Test ===")
    
    # Simple arithmetic
    result = 2 + 2
    print(f"2 + 2 = {result}")
    
    # String operations
    message = "Hello from sandbox!"
    print(f"Message: {message}")
    
    # List operations
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers]
    print(f"Numbers: {numbers}")
    print(f"Squared: {squared}")
    
    return True

def test_system_info():
    """Test system information access."""
    print("\n=== System Information ===")
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"User ID: {os.getuid()}")
    print(f"Process ID: {os.getpid()}")
    
    return True

def test_file_operations():
    """Test file operations within sandbox."""
    print("\n=== File Operations Test ===")
    
    try:
        # Create a test file
        test_file = "test_output.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test file created in the sandbox.\n")
            f.write(f"Timestamp: {time.time()}\n")
        
        print(f"Created file: {test_file}")
        
        # Read the file back
        with open(test_file, 'r') as f:
            content = f.read()
            print(f"File content:\n{content}")
        
        # List files in current directory
        files = os.listdir('.')
        print(f"Files in current directory: {files}")
        
        return True
        
    except Exception as e:
        print(f"File operation error: {e}")
        return False

def test_memory_usage():
    """Test memory usage monitoring."""
    print("\n=== Memory Usage Test ===")
    
    # Create a large list to test memory limits
    try:
        large_list = []
        for i in range(1000000):  # 1 million items
            large_list.append(f"item_{i}")
            if i % 100000 == 0:
                print(f"Created {i} items...")
        
        print(f"Created list with {len(large_list)} items")
        print(f"List size: {sys.getsizeof(large_list)} bytes")
        
        return True
        
    except MemoryError:
        print("Memory limit reached (expected behavior)")
        return True
    except Exception as e:
        print(f"Memory test error: {e}")
        return False

def test_cpu_intensive():
    """Test CPU-intensive operations."""
    print("\n=== CPU Intensive Test ===")
    
    try:
        # Calculate prime numbers
        primes = []
        for num in range(2, 10000):
            is_prime = True
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
                if len(primes) % 100 == 0:
                    print(f"Found {len(primes)} primes...")
        
        print(f"Found {len(primes)} prime numbers")
        return True
        
    except Exception as e:
        print(f"CPU test error: {e}")
        return False

def test_network_access():
    """Test network access (should be blocked in sandbox)."""
    print("\n=== Network Access Test ===")
    
    try:
        import socket
        
        # Try to connect to a remote server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('8.8.8.8', 80))
        sock.close()
        
        if result == 0:
            print("Network access: SUCCESS (unexpected)")
            return False
        else:
            print("Network access: BLOCKED (expected)")
            return True
            
    except Exception as e:
        print(f"Network test error: {e}")
        return True

def main():
    """Main test function."""
    print("🚀 Starting Chroot Lite Sandbox Test")
    print("=" * 50)
    
    tests = [
        ("Basic Operations", test_basic_operations),
        ("System Information", test_system_info),
        ("File Operations", test_file_operations),
        ("Memory Usage", test_memory_usage),
        ("CPU Intensive", test_cpu_intensive),
        ("Network Access", test_network_access)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed (this may be expected behavior)")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
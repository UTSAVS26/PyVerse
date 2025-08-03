#!/usr/bin/env python3
"""
Simple test script to verify PyRecon installation and basic functionality.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import pyrecon
        print("✓ pyrecon package imported successfully")
        
        from pyrecon.core.scanner import PortScanner
        print("✓ PortScanner imported successfully")
        
        from pyrecon.core.banner_grabber import BannerGrabber
        print("✓ BannerGrabber imported successfully")
        
        from pyrecon.core.os_fingerprint import OSFingerprinter
        print("✓ OSFingerprinter imported successfully")
        
        from pyrecon.output.formatter import OutputFormatter
        print("✓ OutputFormatter imported successfully")
        
        from pyrecon.core.utils import parse_target, parse_port_range
        print("✓ Utility functions imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without network calls."""
    print("\nTesting basic functionality...")
    
    try:
        from pyrecon.core.utils import parse_target, parse_port_range, get_service_name
        
        # Test target parsing
        targets = parse_target("192.168.1.1")
        assert targets == ["192.168.1.1"], f"Expected ['192.168.1.1'], got {targets}"
        print("✓ Target parsing works")
        
        # Test port parsing
        ports = parse_port_range("80,443,8080")
        assert ports == [80, 443, 8080], f"Expected [80, 443, 8080], got {ports}"
        print("✓ Port parsing works")
        
        # Test service name
        service = get_service_name(80)
        assert service == "HTTP", f"Expected 'HTTP', got {service}"
        print("✓ Service name lookup works")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

def test_scanner_initialization():
    """Test scanner initialization."""
    print("\nTesting scanner initialization...")
    
    try:
        from pyrecon.core.scanner import PortScanner
        
        scanner = PortScanner(max_workers=10, timeout=1.0)
        assert scanner.max_workers == 10
        assert scanner.timeout == 1.0
        print("✓ Scanner initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Scanner initialization error: {e}")
        return False

def test_formatter_initialization():
    """Test formatter initialization."""
    print("\nTesting formatter initialization...")
    
    try:
        from pyrecon.output.formatter import OutputFormatter
        
        formatter = OutputFormatter(pretty=True, json_output=None)
        assert formatter.pretty == True
        print("✓ Formatter initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Formatter initialization error: {e}")
        return False

def test_cli_import():
    """Test CLI import."""
    print("\nTesting CLI import...")
    
    try:
        from pyrecon.cli.main import cli
        print("✓ CLI import works")
        return True
        
    except Exception as e:
        print(f"✗ CLI import error: {e}")
        return False

def main():
    """Run all tests."""
    print("PyRecon Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_scanner_initialization,
        test_formatter_initialization,
        test_cli_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PyRecon is ready to use.")
        print("\nYou can now run:")
        print("  pyrecon --help")
        print("  pyrecon scan 127.0.0.1 --top-ports 10")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
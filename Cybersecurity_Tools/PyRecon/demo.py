#!/usr/bin/env python3
"""
Demo script for PyRecon - High-Speed Port Scanner & Service Fingerprinter
"""

import sys
import time
from pyrecon.core.scanner import PortScanner
from pyrecon.output.formatter import OutputFormatter
from pyrecon.core.utils import parse_target, parse_port_range, get_service_name

def demo_basic_functionality():
    """Demo basic functionality without network calls."""
    print("🔍 PyRecon Demo - Basic Functionality")
    print("=" * 50)
    
    # Test target parsing
    print("\n1. Target Parsing:")
    targets = parse_target("192.168.1.1")
    print(f"   Single IP: {targets}")
    
    targets = parse_target("192.168.1.0/30")
    print(f"   CIDR Range: {targets}")
    
    # Test port parsing
    print("\n2. Port Parsing:")
    ports = parse_port_range("80,443,8080")
    print(f"   Port list: {ports}")
    
    ports = parse_port_range("80-82")
    print(f"   Port range: {ports}")
    
    ports = parse_port_range("top-5")
    print(f"   Top ports: {ports}")
    
    # Test service names
    print("\n3. Service Names:")
    for port in [80, 443, 22, 21, 25]:
        service = get_service_name(port)
        print(f"   Port {port}: {service}")

def demo_scanner_initialization():
    """Demo scanner initialization."""
    print("\n4. Scanner Initialization:")
    scanner = PortScanner(max_workers=10, timeout=1.0)
    print(f"   Max workers: {scanner.max_workers}")
    print(f"   Timeout: {scanner.timeout}s")
    print(f"   Banner grabber: {scanner.banner_grabber}")
    print(f"   OS fingerprinter: {scanner.os_fingerprinter}")

def demo_formatter():
    """Demo output formatter."""
    print("\n5. Output Formatter:")
    formatter = OutputFormatter(pretty=True)
    print(f"   Pretty output: {formatter.pretty}")
    print(f"   Console: {formatter.console}")

def demo_scan_example():
    """Demo a simple scan example."""
    print("\n6. Scan Example (localhost):")
    print("   Running: scan 127.0.0.1 -p 80,443,22,21")
    
    scanner = PortScanner(max_workers=5, timeout=0.5)
    formatter = OutputFormatter(pretty=False)
    
    start_time = time.time()
    results = scanner.scan("127.0.0.1", "80,443,22,21", fingerprint=False)
    scan_time = time.time() - start_time
    
    print(f"   Scan completed in {scan_time:.2f}s")
    print(f"   Found {len(results)} open ports")
    
    if results:
        for result in results:
            print(f"   - {result.port}/{result.protocol}: {result.service}")
    else:
        print("   - No open ports found (expected for localhost)")

def demo_usage_examples():
    """Show usage examples."""
    print("\n7. Usage Examples:")
    print("   # Quick scan of common ports")
    print("   pyrecon scan 192.168.1.1 --top-ports 100")
    print()
    print("   # Full scan with fingerprinting")
    print("   pyrecon scan example.com -p 1-1024 --fingerprint")
    print()
    print("   # UDP scan")
    print("   pyrecon scan 10.0.0.1 --protocol udp")
    print()
    print("   # Save results to JSON")
    print("   pyrecon scan target.com --json results.json")
    print()
    print("   # Scan multiple targets from file")
    print("   pyrecon scan -f targets.txt --fingerprint")

def main():
    """Run the demo."""
    try:
        demo_basic_functionality()
        demo_scanner_initialization()
        demo_formatter()
        demo_scan_example()
        demo_usage_examples()
        
        print("\n" + "=" * 50)
        print("🎉 PyRecon Demo Completed Successfully!")
        print("The scanner is ready for use.")
        print("Run 'python -c \"from pyrecon.cli.main import cli; cli()\" --help' for CLI help.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
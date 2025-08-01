#!/usr/bin/env python3
"""
Test runner for Behavior-Based Script Detector

This script runs all tests and provides a summary of results.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests and return results."""
    print("🧪 Running Behavior-Based Script Detector Tests")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        try:
            import pytest
        except ImportError:
            print("❌ Failed to install pytest")
            return False
    
    # Run tests
    print("📋 Running test suite...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def test_cli_functionality():
    """Test CLI functionality with example files."""
    print("\n🔧 Testing CLI functionality...")
    
    # Test with malicious file
    print("Testing with malicious example...")
    result = subprocess.run([
        sys.executable, "cli/main.py", "--file", "examples/test_malicious.py", "--no-console"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Malicious file analysis completed")
    else:
        print(f"❌ Malicious file analysis failed: {result.stderr}")
    
    # Test with safe file
    print("Testing with safe example...")
    result = subprocess.run([
        sys.executable, "cli/main.py", "--file", "examples/test_safe.py", "--no-console"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Safe file analysis completed")
    else:
        print(f"❌ Safe file analysis failed: {result.stderr}")
    
    # Test report generation
    print("Testing report generation...")
    result = subprocess.run([
        sys.executable, "cli/main.py", "--file", "examples/test_malicious.py", 
        "--json", "--markdown", "--report", "test_reports/"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Report generation completed")
        # Check if reports were created
        if os.path.exists("test_reports/"):
            reports = list(Path("test_reports/").glob("*.json")) + list(Path("test_reports/").glob("*.md"))
            print(f"📄 Generated {len(reports)} report files")
    else:
        print(f"❌ Report generation failed: {result.stderr}")

def main():
    """Main test runner function."""
    print("🛡️ Behavior-Based Script Detector - Test Suite")
    print("=" * 60)
    
    # Check project structure
    required_files = [
        "analyzer/__init__.py",
        "analyzer/pattern_rules.py",
        "analyzer/score_calculator.py",
        "analyzer/report_generator.py",
        "utils/__init__.py",
        "utils/file_loader.py",
        "cli/__init__.py",
        "cli/main.py",
        "tests/test_patterns.py",
        "examples/test_malicious.py",
        "examples/test_safe.py",
        "requirements.txt",
        "README.md"
    ]
    
    print("📁 Checking project structure...")
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    # Run unit tests
    # Run unit tests
    test_success = run_tests()
    
    # Run CLI tests
    cli_success = test_cli_functionality()
    
    # Combine results
    test_success = test_success and cli_success
    # Summary
    print("\n" + "=" * 60)
    if test_success:
        print("🎉 All tests completed successfully!")
        print("✅ Project is ready for use")
    else:
        print("❌ Some tests failed")
        print("🔧 Please check the output above for issues")
    
    return test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
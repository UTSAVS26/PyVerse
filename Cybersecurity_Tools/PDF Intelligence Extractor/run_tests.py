#!/usr/bin/env python3
"""
Test runner for PDF Intelligence Extractor.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests with coverage."""
    print("🧪 Running PDF Intelligence Extractor Tests")
    print("=" * 50)
    
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "pdf_intelligence_extractor/tests/",
            "-v",
            "--cov=pdf_intelligence_extractor",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Test Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e.stderr}")
        return False

def main():
    """Main function."""
    print("🚀 PDF Intelligence Extractor - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Error: requirements.txt not found. Please run from the project root.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n🎉 Test suite completed successfully!")
        print("📊 Coverage report generated in htmlcov/")
    else:
        print("\n💥 Test suite failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
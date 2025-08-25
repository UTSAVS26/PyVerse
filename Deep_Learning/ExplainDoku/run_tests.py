#!/usr/bin/env python3
"""
Test runner for ExplainDoku
"""

import sys
import subprocess
import os

def run_tests():
    """Run all tests for ExplainDoku"""
    print("🧩 Running ExplainDoku Tests")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
    
    # Run tests with coverage
    test_args = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=explaindoku",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    try:
        subprocess.run(test_args, check=True, cwd=str(script_dir), env=env)
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

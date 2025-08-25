#!/usr/bin/env python3
"""
Test runner for SignalSeekerAI project.
Runs all tests and provides a summary of results.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_tests():
    """Run all tests and return results."""
    print("🧪 Running SignalSeekerAI Tests")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "tests/test_spectrum.py",
        "tests/test_agent.py", 
        "tests/test_training.py",
        "tests/test_visualization.py"
    ]
    
    results = {}
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            continue
            
        print(f"\n📋 Running {test_file}...")
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            output = result.stdout
            error_output = result.stderr
            
            # Count test results
            lines = output.split('\n')
            passed = 0
            failed = 0
            
            for line in lines:
                if 'PASSED' in line:
                    passed += 1
                elif 'FAILED' in line:
                    failed += 1
            
            total_tests += passed + failed
            total_passed += passed
            total_failed += failed
            
            # Store results
            results[test_file] = {
                'passed': passed,
                'failed': failed,
                'output': output,
                'error': error_output,
                'return_code': result.returncode
            }
            
            # Print summary for this file
            if failed == 0:
                print(f"✅ {test_file}: {passed} tests passed, {failed} failed")
            else:
                print(f"❌ {test_file}: {passed} tests passed, {failed} failed")
                print(f"   Error output: {error_output[:200]}...")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file}: Test timed out after 5 minutes")
            results[test_file] = {
                'passed': 0,
                'failed': 0,
                'output': '',
                'error': 'Test timed out',
                'return_code': -1
            }
            total_failed += 1
        except Exception as e:
            print(f"💥 {test_file}: Exception occurred: {e}")
            results[test_file] = {
                'passed': 0,
                'failed': 0,
                'output': '',
                'error': str(e),
                'return_code': -1
            }
            total_failed += 1
    
    return results, total_tests, total_passed, total_failed


def print_summary(results, total_tests, total_passed, total_failed):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_file, result in results.items():
        status = "✅ PASS" if result['failed'] == 0 else "❌ FAIL"
        print(f"  {status} {test_file}: {result['passed']} passed, {result['failed']} failed")
    
    if total_failed > 0:
        print("\n🔍 Failed Tests Details:")
        for test_file, result in results.items():
            if result['failed'] > 0:
                print(f"\n  {test_file}:")
                print(f"    Error: {result['error'][:200]}...")
                if result['output']:
                    # Extract failed test names
                    lines = result['output'].split('\n')
                    failed_tests = [line for line in lines if 'FAILED' in line]
                    for test in failed_tests[:3]:  # Show first 3 failures
                        print(f"    {test.strip()}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'scipy', 
        'matplotlib',
        'seaborn',
        'torch',
        'pytest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def main():
    """Main test runner function."""
    print("🚀 SignalSeekerAI Test Runner")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot run tests due to missing dependencies.")
        sys.exit(1)
    
    # Check if test files exist
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Tests directory not found!")
        print("Please ensure you're running this from the project root directory.")
        sys.exit(1)
    
    # Run tests
    start_time = time.time()
    results, total_tests, total_passed, total_failed = run_tests()
    end_time = time.time()
    
    # Print summary
    print_summary(results, total_tests, total_passed, total_failed)
    
    # Print timing
    duration = end_time - start_time
    print(f"\n⏱️  Total test time: {duration:.2f} seconds")
    
    # Exit with appropriate code
    if total_failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test runner for ArtForgeAI project
"""

import sys
import os
import subprocess
import pytest

def run_tests():
    """Run all tests for the ArtForgeAI project"""
    
    print("🎨 ArtForgeAI - Running Tests")
    print("=" * 50)
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Test files to run
    test_files = [
        "tests/test_canvas.py",
        "tests/test_strokes.py", 
        "tests/test_agent.py",
        "tests/test_train.py"
    ]
    
    # Check if test files exist
    missing_tests = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_tests.append(test_file)
    
    if missing_tests:
        print(f"❌ Missing test files: {missing_tests}")
        return False
    
    # Run tests with pytest
    print("Running tests with pytest...")
    
    try:
        # Run tests with verbose output
        result = pytest.main([
            "--verbose",
            "--tb=short",
            "--disable-warnings",
            "tests/"
        ])
        
        if result == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with exit code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_individual_tests():
    """Run individual test modules"""
    
    print("\n" + "=" * 50)
    print("Running individual test modules...")
    
    test_modules = [
        ("Canvas Tests", "tests.test_canvas"),
        ("Strokes Tests", "tests.test_strokes"),
        ("Agent Tests", "tests.test_agent"),
        ("Train Tests", "tests.test_train")
    ]
    
    all_passed = True
    
    for test_name, test_module in test_modules:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = pytest.main([
                "--verbose",
                "--tb=short",
                "--disable-warnings",
                test_module
            ])
            
            if result == 0:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            all_passed = False
    
    return all_passed

def check_imports():
    """Check if all modules can be imported"""
    
    print("\n" + "=" * 50)
    print("Checking module imports...")
    
    modules = [
        ("canvas", "canvas"),
        ("strokes", "strokes"),
        ("agent", "agent"),
        ("train", "train")
    ]
    
    all_imported = True
    
    for module_name, import_name in modules:
        try:
            __import__(import_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            all_imported = False
        except Exception as e:
            print(f"❌ Error importing {module_name}: {e}")
            all_imported = False
    
    return all_imported

def main():
    """Main test runner function"""
    
    print("🎨 ArtForgeAI Test Suite")
    print("=" * 50)
    
    # Check imports first
    imports_ok = check_imports()
    
    if not imports_ok:
        print("\n❌ Import checks failed. Please fix import issues before running tests.")
        return False
    
    # Run individual tests
    individual_tests_ok = run_individual_tests()
    
    # Run all tests together
    all_tests_ok = run_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if imports_ok and individual_tests_ok and all_tests_ok:
        print("🎉 All tests passed! ArtForgeAI is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

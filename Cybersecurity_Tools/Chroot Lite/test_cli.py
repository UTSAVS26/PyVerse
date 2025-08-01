#!/usr/bin/env python3
"""
Simple test script to verify CLI functionality.
"""

import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from cli.main import ChrootLiteCLI


def test_cli_basic_functionality():
    """Test basic CLI functionality."""
    print("🧪 Testing CLI basic functionality...")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test CLI initialization
        cli = ChrootLiteCLI()
        print("✅ CLI initialization successful")
        
        # Test sandbox creation
        with patch.object(cli.manager, 'create_sandbox') as mock_create:
            mock_create.return_value = True
            cli.create_sandbox("testbox", memory=256, cpu=60)
            mock_create.assert_called_once_with(
                name="testbox",
                memory_limit_mb=256,
                cpu_limit_seconds=60,
                block_network=True
            )
            print("✅ Sandbox creation test passed")
        
        # Test listing sandboxes
        with patch.object(cli.manager, 'list_sandboxes') as mock_list:
            mock_list.return_value = [
                {
                    'name': 'testbox',
                    'memory_limit_mb': 256,
                    'cpu_limit_seconds': 60,
                    'block_network': True,
                    'created_at': '2024-01-01T00:00:00'
                }
            ]
            cli.list_sandboxes()
            print("✅ Sandbox listing test passed")
        
        # Test sandbox deletion
        with patch.object(cli.manager, 'delete_sandbox') as mock_delete:
            mock_delete.return_value = True
            cli.delete_sandbox("testbox")
            mock_delete.assert_called_once_with("testbox")
            print("✅ Sandbox deletion test passed")
        
        # Test cleanup
        with patch.object(cli.manager, 'cleanup_all') as mock_cleanup:
            cli.cleanup_all()
            mock_cleanup.assert_called_once()
            print("✅ Cleanup test passed")
        
        print("🎉 All CLI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_error_handling():
    """Test CLI error handling."""
    print("🧪 Testing CLI error handling...")
    
    cli = ChrootLiteCLI()
    
    # Test creating sandbox that already exists
    with patch.object(cli.manager, 'create_sandbox') as mock_create:
        mock_create.return_value = False
        cli.create_sandbox("existing_box")
        print("✅ Error handling for existing sandbox test passed")
    
    # Test deleting non-existent sandbox
    with patch.object(cli.manager, 'delete_sandbox') as mock_delete:
        mock_delete.return_value = False
        cli.delete_sandbox("nonexistent")
        print("✅ Error handling for non-existent sandbox test passed")
    
    # Test running script in non-existent sandbox
    with patch.object(cli.manager, 'get_sandbox') as mock_get:
        mock_get.return_value = None
        cli.run_script("nonexistent", "script.py")
        print("✅ Error handling for non-existent sandbox execution test passed")
    
    print("🎉 All error handling tests passed!")
    return True


def test_cli_execution():
    """Test CLI execution functionality."""
    print("🧪 Testing CLI execution functionality...")
    
    cli = ChrootLiteCLI()
    
    # Mock executor
    mock_executor = MagicMock()
    mock_executor.execute_script.return_value = {
        'success': True,
        'return_code': 0,
        'execution_time': 1.5,
        'stdout': 'Hello World',
        'stderr': '',
        'pid': 12345
    }
    
    # Test script execution
    with patch.object(cli.manager, 'get_sandbox') as mock_get:
        mock_get.return_value = mock_executor
        cli.run_script("testbox", "script.py", ["arg1", "arg2"])
        mock_executor.execute_script.assert_called_once_with("script.py", ["arg1", "arg2"])
        print("✅ Script execution test passed")
    
    # Test Python code execution
    mock_executor.execute_python_code.return_value = {
        'success': True,
        'return_code': 0,
        'execution_time': 0.5,
        'stdout': 'Hello from Python',
        'stderr': '',
        'pid': 12345
    }
    
    with patch.object(cli.manager, 'get_sandbox') as mock_get:
        mock_get.return_value = mock_executor
        cli.run_python_code("testbox", "print('Hello from Python')")
        mock_executor.execute_python_code.assert_called_once_with("print('Hello from Python')")
        print("✅ Python code execution test passed")
    
    print("🎉 All execution tests passed!")
    return True


def main():
    """Run all CLI tests."""
    print("🚀 Starting CLI Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_cli_basic_functionality),
        ("Error Handling", test_cli_error_handling),
        ("Execution", test_cli_execution)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
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
        print("🎉 All CLI tests passed!")
        return 0
    else:
        print("⚠️ Some CLI tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
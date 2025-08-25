#!/usr/bin/env python3
"""
Test runner script for EcoSimAI project.
Runs all tests and provides a comprehensive summary.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({elapsed_time:.2f}s)")
            print(result.stdout)
            return True, result.stdout, result.stderr
        else:
            print(f"❌ FAILED ({elapsed_time:.2f}s)")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT ({elapsed_time:.2f}s)")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, "", str(e)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'torch', 'matplotlib', 'pygame', 'pytest', 'pytest-cov'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available")
    return True

def run_unit_tests():
    """Run all unit tests."""
    test_files = [
        'test_environment.py',
        'test_agents.py', 
        'test_rl_agent.py',
        'test_simulate.py'
    ]
    
    results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            success, stdout, stderr = run_command(
                f"python -m pytest {test_file} -v",
                f"Unit tests: {test_file}"
            )
            results[test_file] = success
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = False
    
    return results

def run_integration_tests():
    """Run integration tests."""
    success, stdout, stderr = run_command(
        "python -m pytest test_simulate.py::TestSimulationIntegration -v",
        "Integration tests"
    )
    return success

def run_coverage_tests():
    """Run tests with coverage reporting."""
    success, stdout, stderr = run_command(
        "python -m pytest --cov=. --cov-report=term-missing --cov-report=html",
        "Coverage tests"
    )
    return success

def run_quick_simulation_test():
    """Run a quick simulation to test basic functionality."""
    print(f"\n{'='*60}")
    print("Running quick simulation test...")
    print('='*60)
    
    try:
        # Import and run a quick simulation
        from environment import Environment
        from agents import Prey, Predator
        
        # Create small environment
        env = Environment(width=10, height=10)
        
        # Add some agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Run a few steps
        for step in range(5):
            env.step()
            stats = env.get_statistics()
            print(f"Step {step}: Plants={stats['plants']}, Prey={stats['prey']}, Predators={stats['predators']}")
        
        print("✅ Quick simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quick simulation test failed: {e}")
        return False

def run_rl_agent_test():
    """Test RL agent functionality."""
    print(f"\n{'='*60}")
    print("Testing RL agent functionality...")
    print('='*60)
    
    try:
        from environment import Environment
        from rl_agent import RLPrey, RLPredator
        
        # Create environment with RL agents
        env = Environment(width=10, height=10)
        
        # Add RL agents
        rl_prey = RLPrey(0, env._get_random_empty_position(), energy=50)
        rl_predator = RLPredator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(rl_prey, env._get_random_empty_position())
        env._add_agent(rl_predator, env._get_random_empty_position())
        
        # Run a few steps
        for step in range(3):
            env.step()
            stats = env.get_statistics()
            print(f"Step {step}: RL Agents learning...")
        
        # Check that RL agents have memory
        assert len(rl_prey.memory) > 0, "RL Prey should have memory"
        assert len(rl_predator.memory) > 0, "RL Predator should have memory"
        
        print("✅ RL agent test passed")
        return True
        
    except Exception as e:
        print(f"❌ RL agent test failed: {e}")
        return False

def generate_test_report(results):
    """Generate a comprehensive test report."""
    print(f"\n{'='*80}")
    print("📊 TEST REPORT SUMMARY")
    print('='*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if failed_tests == 0:
        print(f"\n🎉 All tests passed! The EcoSimAI project is ready to use.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the output above for details.")
    
    return failed_tests == 0

def main():
    """Main test runner function."""
    print("🌱 EcoSimAI - Test Runner")
    print("="*80)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Initialize results dictionary
    results = {}
    
    # Run different types of tests
    print(f"\n🧪 Running comprehensive test suite...")
    
    # Unit tests
    unit_results = run_unit_tests()
    results.update(unit_results)
    
    # Integration tests
    integration_success = run_integration_tests()
    results['integration_tests'] = integration_success
    
    # Coverage tests
    coverage_success = run_coverage_tests()
    results['coverage_tests'] = coverage_success
    
    # Quick simulation test
    simulation_success = run_quick_simulation_test()
    results['quick_simulation'] = simulation_success
    
    # RL agent test
    rl_success = run_rl_agent_test()
    results['rl_agent_test'] = rl_success
    
    # Generate final report
    all_passed = generate_test_report(results)
    
    # Exit with appropriate code
    if all_passed:
        print(f"\n🚀 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

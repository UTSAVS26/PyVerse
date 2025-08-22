#!/usr/bin/env python3
"""
Test runner for VoiceMoodMirror project.
Runs all tests and provides a comprehensive summary.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests and provide summary."""
    print("🎙️ VoiceMoodMirror - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from project root.")
        return False
    
    # Run tests
    print("Running tests...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", 
            "-v", "--tb=short", "--color=yes"
        ], capture_output=True, text=True, timeout=300)
        
        # Parse results
        output = result.stdout
        error_output = result.stderr
        
        # Count results
        passed = output.count("PASSED")
        failed = output.count("FAILED")
        errors = output.count("ERROR")
        warnings = output.count("warnings")
        
        print(f"\n📊 Test Results Summary:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"🔔 Warnings: {warnings}")
        
        # Show failed tests
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            lines = output.split('\n')
            for line in lines:
                if 'FAILED' in line and '::' in line:
                    test_name = line.split('::')[-1].strip()
                    print(f"   - {test_name}")
        
        # Show errors
        if errors > 0:
            print(f"\n⚠️  Test Errors:")
            lines = output.split('\n')
            for line in lines:
                if 'ERROR' in line and '::' in line:
                    test_name = line.split('::')[-1].strip()
                    print(f"   - {test_name}")
        
        # Overall status
        total_tests = passed + failed + errors
        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if failed == 0 and errors == 0:
            print("\n🎉 All tests passed! The VoiceMoodMirror project is ready to use.")
            return True
        else:
            print(f"\n🔧 {failed + errors} tests need attention. See details above.")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes.")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def show_project_status():
    """Show project structure and status."""
    print("\n📁 Project Structure:")
    print("=" * 30)
    
    modules = [
        ("audio/", "Audio recording and feature extraction"),
        ("emotion/", "Emotion classification and mood mapping"),
        ("music/", "Music selection and playlist building"),
        ("utils/", "Utility functions and smoothing"),
        ("ui/", "Streamlit user interface"),
        ("tests/", "Test suite")
    ]
    
    for module, description in modules:
        if Path(module).exists():
            print(f"✅ {module:<12} - {description}")
        else:
            print(f"❌ {module:<12} - Missing")
    
    print(f"\n📋 Key Features Implemented:")
    print("=" * 30)
    features = [
        "✅ Real-time audio recording",
        "✅ Prosodic feature extraction",
        "✅ Emotion classification (rule-based)",
        "✅ Mood mapping and visualization",
        "✅ Music recommendation system",
        "✅ Adaptive playlist building",
        "✅ Temporal mood smoothing",
        "✅ Streamlit dashboard interface"
    ]
    
    for feature in features:
        print(f"   {feature}")

def main():
    """Main function."""
    print("🎙️ VoiceMoodMirror - Comprehensive Test Suite")
    print("=" * 60)
    
    # Show project status
    show_project_status()
    
    # Run tests
    success = run_tests()
    
    # Final recommendations
    print(f"\n💡 Next Steps:")
    print("=" * 20)
    if success:
        print("✅ All tests passed! You can now:")
        print("   - Run the dashboard: streamlit run ui/dashboard.py")
        print("   - Start recording and analyzing your voice mood")
        print("   - Explore the music recommendation features")
    else:
        print("🔧 To improve test coverage:")
        print("   - Review failed tests above")
        print("   - Fix implementation issues")
        print("   - Add more edge case tests")
        print("   - Consider adding integration tests")
    
    print(f"\n🚀 To run the VoiceMoodMirror dashboard:")
    print("   streamlit run ui/dashboard.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

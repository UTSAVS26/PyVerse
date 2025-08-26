#!/usr/bin/env python3
"""
ArtForgeAI Setup Verification Script
Checks that all components are working correctly
"""

import sys
import os
import importlib

def check_imports():
    """Check that all required modules can be imported"""
    print("🔍 Checking imports...")
    
    required_modules = [
        'numpy',
        'PIL',
        'cv2',
        'torch',
        'matplotlib',
        'pygame',
        'gym',
        'stable_baselines3',
        'pytest'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (missing)")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def check_project_modules():
    """Check that all project modules can be imported"""
    print("\n🔍 Checking project modules...")
    
    project_modules = ['canvas', 'strokes', 'agent', 'train']
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}.py")
        except ImportError as e:
            print(f"  ✗ {module}.py: {e}")
            return False
    
    return True

def check_files():
    """Check that all required files exist"""
    print("\n🔍 Checking project files...")
    
    required_files = [
        'canvas.py',
        'strokes.py', 
        'agent.py',
        'train.py',
        'requirements.txt',
        'README.md',
        'demo.py',
        'quick_test.py',
        'test_basic.py',
        'run_tests.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_directories():
    """Check that required directories exist"""
    print("\n🔍 Checking directories...")
    
    required_dirs = [
        'tests',
        'gallery',
        'gallery/artworks',
        'gallery/models'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0

def check_test_files():
    """Check that test files exist"""
    print("\n🔍 Checking test files...")
    
    test_files = [
        'tests/__init__.py',
        'tests/test_canvas.py',
        'tests/test_strokes.py',
        'tests/test_agent.py',
        'tests/test_train.py'
    ]
    
    missing_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ✓ {test_file}")
        else:
            print(f"  ✗ {test_file} (missing)")
            missing_tests.append(test_file)
    
    return len(missing_tests) == 0

def run_basic_functionality_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Import project modules
        from canvas import Canvas
        from strokes import StrokeGenerator
        from agent import PaintingEnvironment, PaintingAgent
        from train import ArtForgeTrainer
        
        # Test canvas
        canvas = Canvas(100, 100)
        stroke_data = {
            'type': 'line',
            'start_pos': (10, 10),
            'end_pos': (90, 90),
            'color': (255, 0, 0),
            'thickness': 2
        }
        canvas.apply_stroke(stroke_data)
        print("  ✓ Canvas operations")
        
        # Test stroke generator
        generator = StrokeGenerator(100, 100)
        stroke = generator.generate_line_stroke()
        print("  ✓ Stroke generation")
        
        # Test environment
        env = PaintingEnvironment(100, 100, 10)
        state = env.reset()
        print("  ✓ Environment setup")
        
        # Test agent
        agent = PaintingAgent(state.shape[0], 8)
        action = agent.select_action(state, add_noise=False)
        print("  ✓ Agent operations")
        
        # Test trainer
        trainer = ArtForgeTrainer(100, 100, 5)
        stats = trainer.train_episode(render=False, save_artwork=False)
        print("  ✓ Training system")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run complete setup verification"""
    print("🎨 ArtForgeAI Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_imports),
        ("Project Modules", check_project_modules),
        ("Project Files", check_files),
        ("Directories", check_directories),
        ("Test Files", check_test_files),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if check_func():
            passed += 1
            print(f"  ✅ {check_name} - PASSED")
        else:
            print(f"  ❌ {check_name} - FAILED")
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ArtForgeAI is fully set up and ready to use!")
        print("\nNext steps:")
        print("1. Run 'python demo.py' to see the AI in action")
        print("2. Run 'python train.py' to start training")
        print("3. Run 'python quick_test.py' for a quick test")
        print("4. Check the README.md for detailed usage instructions")
    else:
        print(f"\n❌ {total - passed} check(s) failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure all files are in the correct directory")
        print("3. Check Python version compatibility")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

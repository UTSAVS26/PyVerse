#!/usr/bin/env python3
"""Simple import test script."""

print("Starting import test...")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    import scipy
    print("✓ scipy imported")
except Exception as e:
    print(f"✗ scipy import failed: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import torch
    print("✓ torch imported")
except Exception as e:
    print(f"✗ torch import failed: {e}")

try:
    from spectrum import RadioSpectrum
    print("✓ RadioSpectrum imported")
except Exception as e:
    print(f"✗ RadioSpectrum import failed: {e}")

try:
    from agent import RandomAgent
    print("✓ RandomAgent imported")
except Exception as e:
    print(f"✗ RandomAgent import failed: {e}")

try:
    from train import TrainingEnvironment
    print("✓ TrainingEnvironment imported")
except Exception as e:
    print(f"✗ TrainingEnvironment import failed: {e}")

print("Import test completed.")

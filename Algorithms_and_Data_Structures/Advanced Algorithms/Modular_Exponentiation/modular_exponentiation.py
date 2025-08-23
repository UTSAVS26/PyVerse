"""
Modular Exponentiation Algorithms

This module implements various algorithms for computing modular exponentiation,
including binary exponentiation, recursive methods, and Montgomery ladder.

Author: Algorithm Implementation
Date: 2024
"""

import time
import math
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def modular_exponentiation_basic(base: int, exponent: int, modulus: int) -> int:
    """
    Basic modular exponentiation using repeated squaring.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        (base^exponent) mod modulus
    """
    if modulus == 1:
        return 0
    
    result = 1
    base = base % modulus
    
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    
    return result


def modular_exponentiation_recursive(base: int, exponent: int, modulus: int) -> int:
    """
    Recursive modular exponentiation.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        (base^exponent) mod modulus
    """
    if modulus == 1:
        return 0
    
    if exponent == 0:
        return 1
    
    if exponent % 2 == 0:
        half = modular_exponentiation_recursive(base, exponent // 2, modulus)
        return (half * half) % modulus
    else:
        half = modular_exponentiation_recursive(base, exponent // 2, modulus)
        return (base * half * half) % modulus


def modular_exponentiation_montgomery(base: int, exponent: int, modulus: int) -> int:
    """
    Montgomery ladder for modular exponentiation.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        (base^exponent) mod modulus
    """
    if modulus == 1:
        return 0
    
    base = base % modulus
    result = 1
    
    for bit in bin(exponent)[2:]:
        result = (result * result) % modulus
        if bit == '1':
            result = (result * base) % modulus
    
    return result


def modular_exponentiation_binary(base: int, exponent: int, modulus: int) -> int:
    """
    Binary exponentiation with detailed steps.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        (base^exponent) mod modulus
    """
    if modulus == 1:
        return 0
    
    result = 1
    base = base % modulus
    
    # Convert exponent to binary
    binary_exp = bin(exponent)[2:]
    
    for bit in binary_exp:
        result = (result * result) % modulus
        if bit == '1':
            result = (result * base) % modulus
    
    return result


def fermat_primality_test(n: int, k: int = 5) -> bool:
    """
    Fermat primality test using modular exponentiation.
    
    Args:
        n: Number to test
        k: Number of tests
        
    Returns:
        True if probably prime, False if composite
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    for _ in range(k):
        a = np.random.randint(2, n - 1)
        if modular_exponentiation_binary(a, n - 1, n) != 1:
            return False
    
    return True


def miller_rabin_primality_test(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin primality test.
    
    Args:
        n: Number to test
        k: Number of tests
        
    Returns:
        True if probably prime, False if composite
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n as 2^r * d + 1
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for _ in range(k):
        a = np.random.randint(2, n - 1)
        x = modular_exponentiation_binary(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    
    return True


def discrete_logarithm(base: int, result: int, modulus: int) -> Optional[int]:
    """
    Baby-step giant-step algorithm for discrete logarithm.
    
    Args:
        base: Base of logarithm
        result: Result to find logarithm of
        modulus: Modulus
        
    Returns:
        Logarithm if found, None otherwise
    """
    if modulus == 1:
        return None
    
    m = int(math.ceil(math.sqrt(modulus - 1)))
    
    # Baby steps
    baby_steps = {}
    current = 1
    for j in range(m):
        baby_steps[current] = j
        current = (current * base) % modulus
    
    # Giant steps
    factor = modular_exponentiation_binary(base, m * (modulus - 2), modulus)
    current = result
    
    for i in range(m):
        if current in baby_steps:
            return i * m + baby_steps[current]
        current = (current * factor) % modulus
    
    return None


def analyze_performance(base: int, exponent: int, modulus: int) -> dict:
    """
    Analyze performance of different modular exponentiation algorithms.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        Dictionary with performance metrics
    """
    algorithms = [
        ("Basic", modular_exponentiation_basic),
        ("Recursive", modular_exponentiation_recursive),
        ("Montgomery", modular_exponentiation_montgomery),
        ("Binary", modular_exponentiation_binary)
    ]
    
    results = {}
    
    for name, func in algorithms:
        start_time = time.time()
        result = func(base, exponent, modulus)
        execution_time = time.time() - start_time
        
        results[name] = {
            'result': result,
            'time': execution_time
        }
    
    return results


def generate_test_cases() -> List[Tuple[int, int, int]]:
    """
    Generate test cases for modular exponentiation.
    
    Returns:
        List of (base, exponent, modulus) tuples
    """
    test_cases = [
        # Small numbers
        (2, 10, 1000),
        (3, 7, 50),
        (5, 12, 23),
        
        # Medium numbers
        (7, 23, 100),
        (11, 17, 89),
        (13, 19, 97),
        
        # Large numbers
        (2, 100, 1000000007),
        (3, 50, 1000000009),
        (5, 25, 1000000003),
        
        # Edge cases
        (1, 100, 1000),
        (0, 10, 100),
        (10, 0, 100),
        (2, 1000, 1)
    ]
    
    return test_cases


def visualize_performance_comparison(test_cases: List[Tuple[int, int, int]]) -> None:
    """
    Visualize performance comparison of different algorithms.
    
    Args:
        test_cases: List of test cases
    """
    try:
        algorithms = ["Basic", "Recursive", "Montgomery", "Binary"]
        times = {algo: [] for algo in algorithms}
        
        for base, exp, mod in test_cases:
            results = analyze_performance(base, exp, mod)
            for algo in algorithms:
                times[algo].append(results[algo]['time'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        x = range(len(test_cases))
        width = 0.2
        
        for i, algo in enumerate(algorithms):
            plt.bar([x + i * width for x in x], times[algo], 
                   width, label=algo, alpha=0.8)
        
        plt.xlabel('Test Case')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Modular Exponentiation Performance Comparison')
        plt.legend()
        plt.xticks([x + width * 1.5 for x in range(len(test_cases))], 
                   [f'Case {i+1}' for i in range(len(test_cases))])
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib. Install with:")
        print("pip install matplotlib")


def verify_results(base: int, exponent: int, modulus: int) -> bool:
    """
    Verify that all algorithms produce the same result.
    
    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus
        
    Returns:
        True if all results match, False otherwise
    """
    algorithms = [
        modular_exponentiation_basic,
        modular_exponentiation_recursive,
        modular_exponentiation_montgomery,
        modular_exponentiation_binary
    ]
    
    results = [func(base, exponent, modulus) for func in algorithms]
    return len(set(results)) == 1


def rsa_encryption(message: int, e: int, n: int) -> int:
    """
    RSA encryption using modular exponentiation.
    
    Args:
        message: Message to encrypt
        e: Public exponent
        n: Public modulus
        
    Returns:
        Encrypted message
    """
    return modular_exponentiation_binary(message, e, n)


def rsa_decryption(ciphertext: int, d: int, n: int) -> int:
    """
    RSA decryption using modular exponentiation.
    
    Args:
        ciphertext: Encrypted message
        d: Private exponent
        n: Public modulus
        
    Returns:
        Decrypted message
    """
    return modular_exponentiation_binary(ciphertext, d, n)


def main():
    """Main function to demonstrate modular exponentiation algorithms."""
    print("=" * 60)
    print("MODULAR EXPONENTIATION ALGORITHMS")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, (base, exponent, modulus) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Base: {base}, Exponent: {exponent}, Modulus: {modulus}")
        
        # Analyze performance
        results = analyze_performance(base, exponent, modulus)
        
        print("Results:")
        for algo, data in results.items():
            print(f"  {algo}: {data['result']} (Time: {data['time']:.6f}s)")
        
        # Verify results
        is_valid = verify_results(base, exponent, modulus)
        print(f"Verification: {'Valid Valid' if is_valid else 'Invalid Invalid'}")
    
    # RSA example
    print("\n" + "=" * 60)
    print("RSA ENCRYPTION EXAMPLE")
    print("=" * 60)
    
    # Small RSA parameters (for demonstration)
    p, q = 61, 53
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 17  # Public exponent
    d = pow(e, -1, phi)  # Private exponent
    
    message = 123
    print(f"Original message: {message}")
    print(f"Public key: (e={e}, n={n})")
    print(f"Private key: (d={d}, n={n})")
    
    # Encrypt
    encrypted = rsa_encryption(message, e, n)
    print(f"Encrypted: {encrypted}")
    
    # Decrypt
    decrypted = rsa_decryption(encrypted, d, n)
    print(f"Decrypted: {decrypted}")
    print(f"Success: {'Valid Yes' if decrypted == message else 'Invalid No'}")
    
    # Primality testing
    print("\n" + "=" * 60)
    print("PRIMALITY TESTING")
    print("=" * 60)
    
    test_numbers = [2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    print("Fermat Test Results:")
    for num in test_numbers:
        is_prime = fermat_primality_test(num)
        print(f"  {num}: {'Prime' if is_prime else 'Composite'}")
    
    print("\nMiller-Rabin Test Results:")
    for num in test_numbers:
        is_prime = miller_rabin_primality_test(num)
        print(f"  {num}: {'Prime' if is_prime else 'Composite'}")
    
    # Discrete logarithm
    print("\n" + "=" * 60)
    print("DISCRETE LOGARITHM")
    print("=" * 60)
    
    base, modulus = 2, 19
    result = 7
    
    log = discrete_logarithm(base, result, modulus)
    if log is not None:
        print(f"log_{base}({result}) mod {modulus} = {log}")
        verification = modular_exponentiation_binary(base, log, modulus)
        print(f"Verification: {base}^{log} mod {modulus} = {verification}")
    else:
        print(f"No discrete logarithm found for log_{base}({result}) mod {modulus}")
    
    # Performance visualization
    print("\n" + "=" * 60)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    try:
        visualize_performance_comparison(test_cases[:8])  # Use first 8 cases
    except ImportError:
        print("Visualization skipped (matplotlib not available)")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 
"""
Sieve of Eratosthenes Algorithm

An ancient algorithm for finding all prime numbers up to a given limit.
Multiple optimized variants are implemented.
"""

import math
import numpy as np
from typing import List, Set
import time


def sieve_of_eratosthenes_basic(n: int) -> List[int]:
    """
    Basic implementation of Sieve of Eratosthenes.
    
    Args:
        n: Upper limit to find primes up to
        
    Returns:
        List of prime numbers up to n
    """
    if n < 2:
        return []
    
    # Create boolean array for numbers 0 to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve process
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            # Mark all multiples of i as composite
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Collect prime numbers
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    return primes


def sieve_of_eratosthenes_optimized(n: int) -> List[int]:
    """
    Optimized implementation with several improvements.
    
    Args:
        n: Upper limit to find primes up to
        
    Returns:
        List of prime numbers up to n
    """
    if n < 2:
        return []
    
    # Only consider odd numbers (except 2)
    size = (n - 1) // 2
    is_prime = [True] * size
    
    # Sieve process for odd numbers
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if is_prime[(i - 3) // 2]:
            # Mark all odd multiples of i as composite
            start = (i * i - 3) // 2
            for j in range(start, size, i):
                is_prime[j] = False
    
    # Collect prime numbers
    primes = [2]  # 2 is the only even prime
    for i in range(size):
        if is_prime[i]:
            primes.append(2 * i + 3)
    
    return primes


def sieve_of_eratosthenes_numpy(n: int) -> List[int]:
    """
    NumPy-based implementation for better performance.
    
    Args:
        n: Upper limit to find primes up to
        
    Returns:
        List of prime numbers up to n
    """
    if n < 2:
        return []
    
    # Create boolean array
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve process
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            # Mark all multiples of i as composite
            is_prime[i * i::i] = False
    
    # Return prime numbers
    return np.where(is_prime)[0].tolist()


def segmented_sieve(left: int, right: int) -> List[int]:
    """
    Segmented sieve for finding primes in a range.
    
    Args:
        left: Lower bound of range
        right: Upper bound of range
        
    Returns:
        List of prime numbers in [left, right]
    """
    if left > right:
        return []
    
    # Handle edge cases
    if left < 2:
        left = 2
    
    # Find primes up to sqrt(right) for sieving
    limit = int(math.sqrt(right))
    base_primes = sieve_of_eratosthenes_optimized(limit)
    
    # Create boolean array for the range
    size = right - left + 1
    is_prime = [True] * size
    
    # Sieve the range
    for prime in base_primes:
        # Find the first multiple of prime in the range
        first_multiple = max(prime * prime, ((left + prime - 1) // prime) * prime)
        
        # Mark all multiples of prime in the range
        for i in range(first_multiple, right + 1, prime):
            is_prime[i - left] = False
    
    # Collect prime numbers
    primes = []
    for i in range(size):
        if is_prime[i]:
            primes.append(left + i)
    
    return primes


def count_primes(n: int) -> int:
    """
    Count the number of primes up to n using sieve.
    
    Args:
        n: Upper limit
        
    Returns:
        Number of primes up to n
    """
    if n < 2:
        return 0
    
    # Use optimized sieve and count
    primes = sieve_of_eratosthenes_optimized(n)
    return len(primes)


def prime_factors(n: int) -> List[int]:
    """
    Find prime factors of a number using trial division.
    
    Args:
        n: Number to factorize
        
    Returns:
        List of prime factors (with multiplicity)
    """
    if n < 2:
        return []
    
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors


def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def analyze_performance(n: int) -> dict:
    """
    Analyze performance of different sieve implementations.
    
    Args:
        n: Upper limit for prime generation
        
    Returns:
        Dictionary with performance metrics
    """
    results = {}
    
    # Test basic implementation
    start_time = time.time()
    basic_primes = sieve_of_eratosthenes_basic(n)
    basic_time = time.time() - start_time
    results['basic'] = {
        'time': basic_time,
        'count': len(basic_primes),
        'complexity': 'O(n log log n)'
    }
    
    # Test optimized implementation
    start_time = time.time()
    optimized_primes = sieve_of_eratosthenes_optimized(n)
    optimized_time = time.time() - start_time
    results['optimized'] = {
        'time': optimized_time,
        'count': len(optimized_primes),
        'complexity': 'O(n log log n)'
    }
    
    # Test NumPy implementation
    try:
        start_time = time.time()
        numpy_primes = sieve_of_eratosthenes_numpy(n)
        numpy_time = time.time() - start_time
        results['numpy'] = {
            'time': numpy_time,
            'count': len(numpy_primes),
            'complexity': 'O(n log log n)'
        }
    except ImportError:
        results['numpy'] = {
            'time': None,
            'count': None,
            'complexity': 'NumPy not available'
        }
    
    return results


def generate_test_cases() -> List[int]:
    """Generate various test cases for prime generation."""
    return [10, 100, 1000, 10000, 100000]


def visualize_prime_distribution(n: int, show_plot: bool = True):
    """
    Visualize prime number distribution.
    
    Args:
        n: Upper limit
        show_plot: Whether to display the plot
    """
    if not show_plot:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        primes = sieve_of_eratosthenes_optimized(n)
        
        plt.figure(figsize=(12, 8))
        
        # Plot prime numbers
        plt.subplot(2, 1, 1)
        plt.scatter(primes, [1] * len(primes), c='red', s=10, alpha=0.7)
        plt.title(f'Prime Numbers up to {n}')
        plt.xlabel('Number')
        plt.ylabel('Prime')
        plt.grid(True, alpha=0.3)
        
        # Plot prime counting function
        plt.subplot(2, 1, 2)
        x = list(range(2, n + 1))
        y = [len([p for p in primes if p <= i]) for i in x]
        plt.plot(x, y, 'b-', linewidth=1)
        plt.title('Prime Counting Function pi(x)')
        plt.xlabel('x')
        plt.ylabel('pi(x)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


def main():
    """Example usage of Sieve of Eratosthenes algorithm."""
    print("=== Sieve of Eratosthenes Algorithm Demo ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for n in test_cases:
        print(f"Generating primes up to {n}:")
        
        # Performance analysis
        metrics = analyze_performance(n)
        
        for method, data in metrics.items():
            print(f"  {method.title()}:")
            print(f"    Time: {data['time']:.6f}s" if data['time'] else f"    Time: {data['time']}")
            print(f"    Count: {data['count']}")
            print(f"    Complexity: {data['complexity']}")
        
        # Show first few primes for small n
        if n <= 100:
            primes = sieve_of_eratosthenes_optimized(n)
            print(f"  Primes: {primes}")
        
        print()
    
    # Prime factorization example
    print("=== Prime Factorization Examples ===")
    numbers = [12, 100, 1001, 1000000007]
    
    for num in numbers:
        factors = prime_factors(num)
        print(f"{num} = {' × '.join(map(str, factors))}")
    
    print()
    
    # Segmented sieve example
    print("=== Segmented Sieve Example ===")
    left, right = 100, 200
    primes = segmented_sieve(left, right)
    print(f"Primes in [{left}, {right}]: {primes}")
    
    print()
    
    # Visualization example
    print("=== Visualization Example ===")
    visualize_prime_distribution(100, show_plot=True)


if __name__ == "__main__":
    main() 
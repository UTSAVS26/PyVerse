# Sieve of Eratosthenes

An ancient algorithm for finding all prime numbers up to a given limit. Named after the Greek mathematician Eratosthenes of Cyrene.

## Overview

The Sieve of Eratosthenes is one of the most efficient ways to find all primes smaller than a given number n. It works by iteratively marking the multiples of each prime starting from 2.

## Algorithm

1. Create a boolean array `is_prime[0..n]` and initialize all entries as `true`
2. Mark `is_prime[0]` and `is_prime[1]` as `false`
3. For each number `i` from 2 to `√n`:
   - If `is_prime[i]` is `true`, then `i` is prime
   - Mark all multiples of `i` as composite (set `is_prime[j] = false` for all `j` that are multiples of `i`)
4. All remaining `true` values in the array represent prime numbers

## Time Complexity

- **Time**: O(n log log n)
- **Space**: O(n)

## Implementations

### 1. Basic Implementation
- Standard implementation using boolean array
- Marks all multiples of each prime number

### 2. Optimized Implementation
- Only considers odd numbers (except 2)
- Reduces memory usage by half
- More efficient for large numbers

### 3. NumPy Implementation
- Uses NumPy arrays for better performance
- Vectorized operations for faster execution

### 4. Segmented Sieve
- For finding primes in a range [L, R]
- Useful when R is very large but R-L is manageable

## Usage

```python
from sieve_of_eratosthenes import sieve_of_eratosthenes_optimized

# Find all primes up to 100
primes = sieve_of_eratosthenes_optimized(100)
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Count primes up to n
count = len(primes)
print(f"Number of primes up to 100: {count}")  # 25
```

## Additional Functions

### Prime Factorization
```python
from sieve_of_eratosthenes import prime_factors

factors = prime_factors(100)
print(factors)  # [2, 2, 5, 5]
```

### Primality Testing
```python
from sieve_of_eratosthenes import is_prime

print(is_prime(17))  # True
print(is_prime(25))  # False
```

### Segmented Sieve
```python
from sieve_of_eratosthenes import segmented_sieve

# Find primes in range [100, 200]
primes = segmented_sieve(100, 200)
print(primes)
```

## Applications

- **Cryptography**: Prime number generation for RSA, Diffie-Hellman
- **Number Theory**: Mathematical research and proofs
- **Computer Science**: Hash functions, random number generation
- **Competitive Programming**: Efficient prime number generation

## Performance Analysis

The algorithm includes performance analysis functions to compare different implementations:

```python
from sieve_of_eratosthenes import analyze_performance

metrics = analyze_performance(100000)
for method, data in metrics.items():
    print(f"{method}: {data['time']:.6f}s, {data['count']} primes")
```

## Visualization

The implementation includes visualization capabilities:

```python
from sieve_of_eratosthenes import visualize_prime_distribution

# Visualize prime distribution up to 100
visualize_prime_distribution(100, show_plot=True)
```

## Requirements

- Python 3.7+
- NumPy (for NumPy implementation, optional)
- Matplotlib (for visualization, optional)

## Installation

```bash
pip install numpy matplotlib
```

## Mathematical Background

The algorithm is based on the principle that if a number is composite, it must have a prime factor less than or equal to its square root. This allows us to efficiently mark all composite numbers by only checking multiples of primes up to √n.

## Historical Context

The algorithm was first described by Eratosthenes of Cyrene (276-194 BCE), a Greek mathematician, geographer, and astronomer. It remains one of the most efficient algorithms for generating prime numbers and is still widely used in modern computer science. 
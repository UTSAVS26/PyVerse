# Chinese Remainder Theorem

An algorithm to solve systems of simultaneous congruences. A fundamental result in number theory with applications in cryptography and computer algebra.

## Overview

The Chinese Remainder Theorem (CRT) provides a method to solve systems of simultaneous linear congruences. Given a system of congruences with pairwise coprime moduli, CRT guarantees a unique solution modulo the product of the moduli.

## Mathematical Background

### Statement
Given a system of congruences:
- x ≡ a₁ (mod m₁)
- x ≡ a₂ (mod m₂)
- ...
- x ≡ aₖ (mod mₖ)

Where m₁, m₂, ..., mₖ are pairwise coprime, there exists a unique solution modulo M = m₁ × m₂ × ... × mₖ.

### Solution Formula
The solution is given by:
**x = Σ(aᵢ × Mᵢ × Mᵢ⁻¹) mod M**

Where:
- M = m₁ × m₂ × ... × mₖ
- Mᵢ = M/mᵢ
- Mᵢ⁻¹ is the modular inverse of Mᵢ modulo mᵢ

## Algorithm

### Standard CRT Algorithm
1. **Calculate M** = product of all moduli
2. **For each congruence**:
   - Calculate Mᵢ = M/mᵢ
   - Find modular inverse Mᵢ⁻¹ of Mᵢ modulo mᵢ
   - Add aᵢ × Mᵢ × Mᵢ⁻¹ to the sum
3. **Return** the sum modulo M

### Garner's Algorithm
For pairwise coprime moduli, Garner's algorithm provides an efficient way to reconstruct the solution:
1. **Initialize** x = a₁
2. **For each subsequent congruence**:
   - Solve: x + k × m₁ × m₂ × ... × mᵢ₋₁ ≡ aᵢ (mod mᵢ)
   - Update x = x + k × m₁ × m₂ × ... × mᵢ₋₁
3. **Return** the final x

## Time Complexity

- **Standard CRT**: O(k log n) where k is number of congruences, n is max modulus
- **Garner's Algorithm**: O(k² log n)
- **Space**: O(k)

## Implementations

### 1. Standard CRT
- Iterative approach using modular inverses
- Handles any number of congruences
- Requires pairwise coprime moduli

### 2. Garner's Algorithm
- More efficient for large numbers
- Requires pairwise coprime moduli
- Better for cryptographic applications

### 3. All Solutions
- Finds particular solution and period
- Handles non-coprime moduli
- Returns solution and period for all solutions

## Usage

```python
from chinese_remainder_theorem import chinese_remainder_theorem

# Basic usage
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = chinese_remainder_theorem(remainders, moduli)
print(solution)  # 23

# Verify solution
for r, m in zip(remainders, moduli):
    print(f"{solution} ≡ {solution % m} (mod {m})")
```

## Applications

### 1. Cryptography
```python
# RSA decryption using CRT
def rsa_decrypt_crt(ciphertext, d, p, q):
    # Decrypt modulo p and q separately
    m1 = pow(ciphertext, d % (p-1), p)
    m2 = pow(ciphertext, d % (q-1), q)
    
    # Use CRT to combine results
    remainders = [m1, m2]
    moduli = [p, q]
    return chinese_remainder_theorem(remainders, moduli)
```

### 2. Computer Algebra
```python
# Large integer arithmetic
def large_mod_arithmetic(number, moduli):
    remainders = [number % m for m in moduli]
    return chinese_remainder_theorem(remainders, moduli)
```

### 3. Error Correction
```python
# Reed-Solomon decoding
def reed_solomon_decode(received_values, evaluation_points):
    # Use CRT to reconstruct the polynomial
    return chinese_remainder_theorem(received_values, evaluation_points)
```

## Mathematical Examples

### Example 1: Simple System
Solve:
- x ≡ 2 (mod 3)
- x ≡ 3 (mod 5)
- x ≡ 2 (mod 7)

**Solution**: x = 23
- 23 ≡ 2 (mod 3) ✓
- 23 ≡ 3 (mod 5) ✓
- 23 ≡ 2 (mod 7) ✓

### Example 2: Large Numbers
Solve:
- x ≡ 123456 (mod 1000000007)
- x ≡ 987654 (mod 1000000009)

**Solution**: x = 1234567890123456789

## Performance Analysis

The implementation includes performance comparison:

```python
from chinese_remainder_theorem import analyze_performance

metrics = analyze_performance(remainders, moduli)
print(f"Number of congruences: {metrics['num_congruences']}")
print(f"Standard CRT time: {metrics['standard_time']:.6f}s")
print(f"Garner's time: {metrics['garner_time']:.6f}s")
print(f"Solution: {metrics['solution']}")
```

## Edge Cases

### Non-Coprime Moduli
- Standard CRT fails if moduli are not pairwise coprime
- All solutions function handles this case
- Returns solution and period for all solutions

### Empty System
- Returns None for empty input
- Handles single congruence case

### Large Numbers
- Uses extended Euclidean algorithm for modular inverses
- Handles overflow with proper modular arithmetic
- Efficient for cryptographic applications

## Comparison with Other Methods

| Method | Time Complexity | Space Complexity | Advantages |
|--------|----------------|------------------|------------|
| Standard CRT | O(k log n) | O(k) | Simple, handles any number of congruences |
| Garner's Algorithm | O(k² log n) | O(k) | Efficient for large numbers |
| All Solutions | O(k log n) | O(k) | Handles non-coprime moduli |
| Naive Search | O(M) | O(1) | Simple but inefficient |

## Historical Context

The Chinese Remainder Theorem was first described in the 3rd century CE by the Chinese mathematician Sun Tzu in his work "Sun Tzu Suan Ching" (The Mathematical Classic of Sun Tzu). The theorem was later rediscovered by mathematicians in Europe and has become a fundamental result in number theory and cryptography.

## Applications in Computer Science

### 1. Cryptography
- **RSA Cryptography**: CRT-RSA for efficient decryption
- **Secret Sharing**: Shamir's Secret Sharing Scheme
- **Digital Signatures**: CRT-based signature schemes
- **Elliptic Curve Cryptography**: Point multiplication

### 2. Computer Algebra
- **Polynomial Interpolation**: Lagrange interpolation
- **Large Integer Arithmetic**: Modular arithmetic with large numbers
- **Symbolic Computation**: Computer algebra systems
- **Error Correction**: Reed-Solomon codes

### 3. Algorithm Design
- **Parallel Computing**: Divide-and-conquer algorithms
- **Distributed Systems**: Consensus algorithms
- **Database Systems**: Hash functions and indexing
- **Network Protocols**: Error detection and correction

## Requirements

- Python 3.7+
- No external dependencies required

## Installation

No additional installation required beyond Python standard library.

## Implementation Details

### Modular Inverse Calculation
```python
def mod_inverse(a, m):
    """Calculate modular multiplicative inverse using extended Euclidean algorithm."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x
    
    gcd_val, x, y = extended_gcd(a, m)
    if gcd_val != 1:
        return None  # Modular inverse doesn't exist
    
    return (x % m + m) % m
```

### Standard CRT Implementation
```python
def chinese_remainder_theorem(remainders, moduli):
    """Solve system of congruences using Chinese Remainder Theorem."""
    if len(remainders) != len(moduli):
        return None
    
    if not remainders:
        return None
    
    # Start with first congruence
    result = remainders[0]
    current_modulus = moduli[0]
    
    # Iteratively solve each congruence
    for i in range(1, len(remainders)):
        a = current_modulus
        b = moduli[i]
        c = remainders[i] - result
        
        # Solve: ax ≡ c (mod b)
        solution = solve_linear_congruence(a, c, b)
        
        if solution is None:
            return None  # No solution exists
        
        x0, period = solution
        result += x0 * current_modulus
        current_modulus = (current_modulus * moduli[i]) // gcd(current_modulus, moduli[i])
    
    return result % current_modulus
```

## Verification

The implementation includes verification functions:

```python
from chinese_remainder_theorem import verify_crt_solution

# Verify solution
is_valid = verify_crt_solution(solution, remainders, moduli)
print(f"Solution is valid: {is_valid}")
```

## Advanced Topics

### Non-Coprime Moduli
When moduli are not pairwise coprime, the system may have:
- **No solution**: If there's a contradiction
- **Multiple solutions**: All solutions differ by the LCM of the moduli
- **Unique solution**: If the system is consistent

### Large Number Optimization
For very large numbers:
- Use Garner's algorithm for efficiency
- Implement Montgomery multiplication
- Use specialized libraries for cryptographic applications

### Parallel Implementation
For systems with many congruences:
- Process congruences in parallel
- Use divide-and-conquer approach
- Implement distributed CRT algorithms 
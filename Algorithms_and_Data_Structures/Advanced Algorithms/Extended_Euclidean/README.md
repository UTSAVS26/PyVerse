# Extended Euclidean Algorithm

An extension of the Euclidean algorithm that finds Bézout coefficients in addition to the greatest common divisor.

## Overview

The Extended Euclidean Algorithm not only computes the GCD of two integers but also finds integers x and y such that:

**ax + by = gcd(a, b)**

This is known as Bézout's identity, and the coefficients x and y are called Bézout coefficients.

## Algorithm

The algorithm extends the standard Euclidean algorithm by keeping track of the coefficients:

1. Initialize: `old_r = a`, `r = b`, `old_s = 1`, `s = 0`, `old_t = 0`, `t = 1`
2. While `r ≠ 0`:
   - `quotient = old_r // r`
   - `old_r, r = r, old_r - quotient * r`
   - `old_s, s = s, old_s - quotient * s`
   - `old_t, t = t, old_t - quotient * t`
3. Return `(old_r, old_s, old_t)` where `old_r` is the GCD

## Time Complexity

- **Time**: O(log min(a, b))
- **Space**: O(1)

## Applications

### 1. Modular Multiplicative Inverse
```python
from extended_euclidean import mod_inverse

# Find modular inverse of a modulo m
inverse = mod_inverse(3, 7)  # Returns 5
# Verification: 3 * 5 ≡ 1 (mod 7)
```

### 2. Linear Congruence Solving
```python
from extended_euclidean import solve_linear_congruence

# Solve: 3x ≡ 2 (mod 7)
solution = solve_linear_congruence(3, 2, 7)
# Returns (3, 7) meaning x ≡ 3 (mod 7)
```

### 3. Linear Diophantine Equations
```python
from extended_euclidean import solve_diophantine_equation

# Solve: 3x + 5y = 1
solution = solve_diophantine_equation(3, 5, 1)
# Returns (2, -1) meaning x = 2, y = -1
```

### 4. Chinese Remainder Theorem
```python
from extended_euclidean import chinese_remainder_theorem

# Solve system of congruences
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = chinese_remainder_theorem(remainders, moduli)
# Returns 23 (unique solution modulo 105)
```

## Mathematical Background

### Bézout's Identity
For any integers a and b, there exist integers x and y such that:
**ax + by = gcd(a, b)**

### Linear Congruence
A linear congruence has the form: **ax ≡ b (mod m)**
- Has a solution if and only if gcd(a, m) divides b
- If a solution exists, there are exactly gcd(a, m) solutions modulo m

### Linear Diophantine Equation
A linear Diophantine equation has the form: **ax + by = c**
- Has a solution if and only if gcd(a, b) divides c
- If (x₀, y₀) is one solution, all solutions are:
  - x = x₀ + (b/gcd(a,b))t
  - y = y₀ - (a/gcd(a,b))t
  where t is any integer

## Usage Examples

### Basic Extended GCD
```python
from extended_euclidean import extended_gcd

gcd_val, x, y = extended_gcd(48, 18)
print(f"GCD(48, 18) = {gcd_val}")
print(f"48 × {x} + 18 × {y} = {48 * x + 18 * y}")
```

### Modular Inverse
```python
from extended_euclidean import mod_inverse

# Find inverse of 3 modulo 7
inverse = mod_inverse(3, 7)
if inverse is not None:
    print(f"3^(-1) mod 7 = {inverse}")
    print(f"Verification: 3 × {inverse} ≡ {(3 * inverse) % 7} (mod 7)")
else:
    print("Modular inverse does not exist")
```

### Linear Congruence
```python
from extended_euclidean import solve_linear_congruence

# Solve 5x ≡ 3 (mod 12)
solution = solve_linear_congruence(5, 3, 12)
if solution is not None:
    x0, period = solution
    print(f"Particular solution: x = {x0}")
    print(f"All solutions: x ≡ {x0} (mod {period})")
else:
    print("No solution exists")
```

## Requirements

- Python 3.7+
- No external dependencies required

## Implementation Details

The implementation handles:
- **Negative numbers**: Automatically converts to positive for computation
- **Zero inputs**: Properly handles edge cases
- **Sign preservation**: Maintains correct signs in Bézout coefficients
- **Modular arithmetic**: Ensures results are in the correct range

## Performance Analysis

The algorithm includes performance analysis functions:

```python
from extended_euclidean import analyze_performance

metrics = analyze_performance(48, 18)
print(f"Execution time: {metrics['execution_time']:.6f}s")
print(f"Verification: {metrics['verification']}")
print(f"Correct: {metrics['correct']}")
```

## Historical Context

The Extended Euclidean Algorithm is a natural extension of the Euclidean algorithm, which was first described by Euclid in his "Elements" around 300 BCE. The extended version was developed to solve problems in number theory and has become fundamental in modern cryptography and computer algebra systems.

## Applications in Cryptography

- **RSA Cryptography**: Computing private key components
- **Diffie-Hellman**: Key exchange protocols
- **Elliptic Curve Cryptography**: Point operations
- **Digital Signatures**: Signature generation and verification 
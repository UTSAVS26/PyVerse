# Modular Exponentiation

Efficient algorithm for computing large powers modulo n. Also known as "fast exponentiation" or "binary exponentiation".

## Overview

Modular exponentiation is the computation of the remainder when an integer b (the base) is raised to the eth power (the exponent), and divided by a positive integer m (the modulus). In symbols, given base b, exponent e, and modulus m, the modular exponentiation c is:

**c ≡ b^e (mod m)**

## Algorithm

### Binary Exponentiation (Fast Exponentiation)

The most efficient method uses the binary representation of the exponent:

1. Initialize result = 1
2. For each bit in the binary representation of the exponent (from least to most significant):
   - If the bit is 1, multiply result by base (mod modulus)
   - Square the base (mod modulus)
3. Return the final result

### Example
To compute 3^13 mod 7:
- 13 in binary is 1101
- Start with result = 1, base = 3
- Bit 0 (1): result = (1 × 3) mod 7 = 3
- Bit 1 (0): result = 3, base = (3²) mod 7 = 2
- Bit 2 (1): result = (3 × 2) mod 7 = 6
- Bit 3 (1): result = (6 × 4) mod 7 = 3

## Time Complexity

- **Time**: O(log exponent)
- **Space**: O(1)

## Implementations

### 1. Basic Implementation
- Uses repeated multiplication
- Simple but inefficient for large exponents
- Time complexity: O(exponent)

### 2. Binary Exponentiation
- Most commonly used method
- Uses binary representation of exponent
- Time complexity: O(log exponent)

### 3. Recursive Implementation
- Recursive version of binary exponentiation
- Same time complexity as iterative version
- More elegant but uses stack space

### 4. Montgomery Ladder
- Provides protection against timing attacks
- Constant-time execution regardless of exponent
- Important for cryptographic applications

## Usage

```python
from modular_exponentiation import mod_pow

# Basic usage
result = mod_pow(2, 10, 1000)  # 2^10 mod 1000
print(result)  # 24

# Large numbers
result = mod_pow(3, 1000000, 1000000007)
print(result)

# Different methods
result_binary = mod_pow(5, 1000, 1000000007, method="binary")
result_recursive = mod_pow(5, 1000, 1000000007, method="recursive")
result_montgomery = mod_pow(5, 1000, 1000000007, method="montgomery")
```

## Applications

### 1. RSA Cryptography
```python
# RSA encryption: c = m^e mod n
def rsa_encrypt(message, e, n):
    return mod_pow(message, e, n)

# RSA decryption: m = c^d mod n
def rsa_decrypt(ciphertext, d, n):
    return mod_pow(ciphertext, d, n)
```

### 2. Primality Testing
```python
from modular_exponentiation import fermat_little_theorem_test, miller_rabin_test

# Fermat's Little Theorem test
is_prime = fermat_little_theorem_test(17, k=5)

# Miller-Rabin test (more reliable)
is_prime = miller_rabin_test(17, k=5)
```

### 3. Discrete Logarithm
```python
# Find discrete logarithm: solve a^x ≡ b (mod p)
def discrete_log(a, b, p):
    for x in range(p):
        if mod_pow(a, x, p) == b:
            return x
    return None
```

## Mathematical Background

### Fermat's Little Theorem
If p is prime and a is not divisible by p, then:
**a^(p-1) ≡ 1 (mod p)**

### Euler's Theorem
If gcd(a, n) = 1, then:
**a^φ(n) ≡ 1 (mod n)**
where φ(n) is Euler's totient function.

### Chinese Remainder Theorem
For coprime moduli m₁, m₂, ..., mₖ:
**x ≡ a₁ (mod m₁), x ≡ a₂ (mod m₂), ..., x ≡ aₖ (mod mₖ)**
has a unique solution modulo m₁m₂...mₖ.

## Performance Analysis

The implementation includes performance comparison functions:

```python
from modular_exponentiation import analyze_performance

metrics = analyze_performance(2, 1000, 1000000007)
for method, data in metrics.items():
    if data['success']:
        print(f"{method}: {data['time']:.6f}s")
    else:
        print(f"{method}: {data['error']}")
```

## Security Considerations

### Timing Attacks
- Basic implementations may leak information through timing
- Montgomery ladder provides constant-time execution
- Important for cryptographic applications

### Side-Channel Attacks
- Power analysis attacks can reveal the exponent
- Montgomery ladder helps mitigate these attacks
- Additional countermeasures may be needed

## Requirements

- Python 3.7+
- No external dependencies required

## Installation

No additional installation required beyond Python standard library.

## Historical Context

The binary exponentiation algorithm was first described by the Indian mathematician Pingala in the 2nd century BCE. The modern form was developed for computer applications and has become fundamental in cryptography and computer algebra systems.

## Applications in Cryptography

- **RSA Cryptography**: Encryption and decryption
- **Diffie-Hellman**: Key exchange
- **Digital Signatures**: DSA, ECDSA
- **Elliptic Curve Cryptography**: Point multiplication
- **Zero-Knowledge Proofs**: Commitment schemes 
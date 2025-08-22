"""
Extended Euclidean Algorithm

Extended version of the Euclidean algorithm to find Bézout coefficients.
Used for solving linear Diophantine equations and modular arithmetic.
"""

import math
from typing import Tuple, Optional
import time


def gcd(a: int, b: int) -> int:
    """
    Calculate Greatest Common Divisor using Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        GCD of a and b
    """
    while b:
        a, b = b, a % b
    return abs(a)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.
    
    Finds integers x, y such that: ax + by = gcd(a, b)
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple (gcd, x, y) where gcd is the GCD of a and b,
        and x, y are Bézout coefficients
    """
    if a == 0:
        return abs(b), 0, 1 if b > 0 else -1
    
    if b == 0:
        return abs(a), 1 if a > 0 else -1, 0
    
    # Ensure positive numbers for the algorithm
    sign_a = 1 if a > 0 else -1
    sign_b = 1 if b > 0 else -1
    a, b = abs(a), abs(b)
    
    # Extended Euclidean algorithm
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    # Apply original signs
    return old_r, old_s * sign_a, old_t * sign_b


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Find modular multiplicative inverse of a modulo m.
    
    Args:
        a: Integer
        m: Modulus
        
    Returns:
        Modular inverse if it exists, None otherwise
    """
    if m <= 0:
        return None
    
    gcd_val, x, y = extended_gcd(a, m)
    
    if gcd_val != 1:
        return None  # Modular inverse doesn't exist
    
    # Ensure result is positive
    return (x % m + m) % m


def solve_linear_congruence(a: int, b: int, m: int) -> Optional[Tuple[int, int]]:
    """
    Solve linear congruence: ax = b (mod m)
    
    Args:
        a: Coefficient of x
        b: Right-hand side
        m: Modulus
        
    Returns:
        Tuple (x0, d) where x0 is a particular solution and d is the period,
        or None if no solution exists
    """
    if m <= 0:
        return None
    
    gcd_val, x, y = extended_gcd(a, m)
    
    if b % gcd_val != 0:
        return None  # No solution exists
    
    # Find particular solution
    x0 = (x * (b // gcd_val)) % m
    
    # Period is m // gcd_val
    period = m // gcd_val
    
    return x0, period


def solve_diophantine_equation(a: int, b: int, c: int) -> Optional[Tuple[int, int]]:
    """
    Solve linear Diophantine equation: ax + by = c
    
    Args:
        a: Coefficient of x
        b: Coefficient of y
        c: Right-hand side
        
    Returns:
        Tuple (x, y) representing a particular solution,
        or None if no solution exists
    """
    gcd_val, x, y = extended_gcd(a, b)
    
    if c % gcd_val != 0:
        return None  # No solution exists
    
    # Find particular solution
    x0 = x * (c // gcd_val)
    y0 = y * (c // gcd_val)
    
    return x0, y0


def chinese_remainder_theorem(remainders: list, moduli: list) -> Optional[int]:
    """
    Solve system of congruences using Chinese Remainder Theorem.
    
    Args:
        remainders: List of remainders
        moduli: List of moduli
        
    Returns:
        Solution to the system of congruences, or None if no solution exists
    """
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
        
        # Solve: ax = c (mod b)
        solution = solve_linear_congruence(a, c, b)
        
        if solution is None:
            return None  # No solution exists
        
        x0, period = solution
        result += x0 * current_modulus
        current_modulus = (current_modulus * moduli[i]) // gcd(current_modulus, moduli[i])
    
    return result % current_modulus


def bezout_identity(a: int, b: int) -> Tuple[int, int, int]:
    """
    Find Bézout coefficients for two integers.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple (gcd, x, y) where ax + by = gcd(a, b)
    """
    return extended_gcd(a, b)


def lcm(a: int, b: int) -> int:
    """
    Calculate Least Common Multiple using GCD.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        LCM of a and b
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def analyze_performance(a: int, b: int) -> dict:
    """
    Analyze performance of extended Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Dictionary with performance metrics
    """
    import time
    
    # Time the algorithm
    start_time = time.time()
    gcd_val, x, y = extended_gcd(a, b)
    end_time = time.time()
    
    # Verify the result
    verification = a * x + b * y
    
    return {
        'a': a,
        'b': b,
        'gcd': gcd_val,
        'x': x,
        'y': y,
        'execution_time': end_time - start_time,
        'verification': verification,
        'correct': verification == gcd_val
    }


def generate_test_cases() -> list:
    """Generate various test cases for extended Euclidean algorithm."""
    return [
        (48, 18),
        (100, 35),
        (12345, 67890),
        (1, 1),
        (0, 5),
        (5, 0),
        (-48, 18),
        (48, -18),
    ]


def main():
    """Example usage of Extended Euclidean Algorithm."""
    print("=== Extended Euclidean Algorithm Demo ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for a, b in test_cases:
        print(f"Extended GCD({a}, {b}):")
        
        # Calculate extended GCD
        gcd_val, x, y = extended_gcd(a, b)
        print(f"  GCD: {gcd_val}")
        print(f"  Bézout coefficients: x = {x}, y = {y}")
        print(f"  Verification: {a} × {x} + {b} × {y} = {a * x + b * y}")
        
        # Performance analysis
        metrics = analyze_performance(a, b)
        print(f"  Execution time: {metrics['execution_time']:.6f}s")
        print(f"  Correct: {metrics['correct']}")
        print()
    
    # Modular inverse examples
    print("=== Modular Inverse Examples ===")
    inverse_cases = [(3, 7), (5, 12), (7, 15), (2, 4)]
    
    for a, m in inverse_cases:
        inverse = mod_inverse(a, m)
        if inverse is not None:
            print(f"Modular inverse of {a} modulo {m}: {inverse}")
            print(f"  Verification: {a} × {inverse} = {(a * inverse) % m} (mod {m})")
        else:
            print(f"Modular inverse of {a} modulo {m}: Does not exist")
        print()
    
    # Linear congruence examples
    print("=== Linear Congruence Examples ===")
    congruence_cases = [(3, 2, 7), (5, 3, 12), (2, 1, 4)]
    
    for a, b, m in congruence_cases:
        solution = solve_linear_congruence(a, b, m)
        if solution is not None:
            x0, period = solution
            print(f"Solution to {a}x = {b} (mod {m}):")
            print(f"  Particular solution: x = {x0}")
            print(f"  Period: {period}")
            print(f"  All solutions: x = {x0} (mod {period})")
        else:
            print(f"No solution to {a}x = {b} (mod {m})")
        print()
    
    # Diophantine equation examples
    print("=== Diophantine Equation Examples ===")
    diophantine_cases = [(3, 5, 1), (6, 9, 3), (2, 4, 1)]
    
    for a, b, c in diophantine_cases:
        solution = solve_diophantine_equation(a, b, c)
        if solution is not None:
            x, y = solution
            print(f"Solution to {a}x + {b}y = {c}:")
            print(f"  x = {x}, y = {y}")
            print(f"  Verification: {a} × {x} + {b} × {y} = {a * x + b * y}")
        else:
            print(f"No solution to {a}x + {b}y = {c}")
        print()
    
    # Chinese Remainder Theorem example
    print("=== Chinese Remainder Theorem Example ===")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    
    solution = chinese_remainder_theorem(remainders, moduli)
    if solution is not None:
        print(f"System of congruences:")
        for i, (r, m) in enumerate(zip(remainders, moduli)):
            print(f"  x = {r} (mod {m})")
        print(f"Solution: x = {solution}")
        
        # Verify solution
        for r, m in zip(remainders, moduli):
            print(f"  {solution} = {solution % m} (mod {m})")
    else:
        print("No solution exists for the given system of congruences")


if __name__ == "__main__":
    main() 
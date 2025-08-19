"""
Chinese Remainder Theorem (CRT)

Algorithm to solve systems of simultaneous congruences.
Used in cryptography, number theory, and computer algebra.
"""

import math
from typing import List, Optional, Tuple
import time


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple (gcd, x, y) where ax + by = gcd(a, b)
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


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> Optional[int]:
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


def chinese_remainder_theorem_garner(remainders: List[int], moduli: List[int]) -> Optional[int]:
    """
    Garner's algorithm for Chinese Remainder Theorem.
    More efficient when moduli are pairwise coprime.
    
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
    
    n = len(remainders)
    
    # Check if moduli are pairwise coprime
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(moduli[i], moduli[j]) != 1:
                return None  # Garner's algorithm requires pairwise coprime moduli
    
    # Garner's algorithm
    result = remainders[0]
    current_modulus = moduli[0]
    
    for i in range(1, n):
        # Find modular inverse
        inv = mod_inverse(current_modulus, moduli[i])
        if inv is None:
            return None
        
        # Update result
        result = (result + (remainders[i] - result) * inv * current_modulus) % (current_modulus * moduli[i])
        current_modulus *= moduli[i]
    
    return result


def chinese_remainder_theorem_all_solutions(remainders: List[int], moduli: List[int]) -> Optional[Tuple[int, int]]:
    """
    Find all solutions to a system of congruences.
    
    Args:
        remainders: List of remainders
        moduli: List of moduli
        
    Returns:
        Tuple (x0, period) where x0 is a particular solution and period is the period,
        or None if no solution exists
    """
    if len(remainders) != len(moduli):
        return None
    
    if not remainders:
        return None
    
    # Find particular solution
    solution = chinese_remainder_theorem(remainders, moduli)
    if solution is None:
        return None
    
    # Calculate period (LCM of all moduli)
    period = 1
    for modulus in moduli:
        period = lcm(period, modulus)
    
    return solution, period


def verify_crt_solution(solution: int, remainders: List[int], moduli: List[int]) -> bool:
    """
    Verify that a solution satisfies all congruences.
    
    Args:
        solution: Proposed solution
        remainders: List of remainders
        moduli: List of moduli
        
    Returns:
        True if solution is correct, False otherwise
    """
    for remainder, modulus in zip(remainders, moduli):
        if solution % modulus != remainder:
            return False
    return True


def analyze_performance(remainders: List[int], moduli: List[int]) -> dict:
    """
    Analyze performance of different CRT implementations.
    
    Args:
        remainders: List of remainders
        moduli: List of moduli
        
    Returns:
        Dictionary with performance metrics
    """
    results = {}
    
    # Test standard CRT
    start_time = time.time()
    solution1 = chinese_remainder_theorem(remainders, moduli)
    time1 = time.time() - start_time
    
    results['standard'] = {
        'solution': solution1,
        'time': time1,
        'success': solution1 is not None
    }
    
    # Test Garner's algorithm
    start_time = time.time()
    solution2 = chinese_remainder_theorem_garner(remainders, moduli)
    time2 = time.time() - start_time
    
    results['garner'] = {
        'solution': solution2,
        'time': time2,
        'success': solution2 is not None
    }
    
    # Test all solutions
    start_time = time.time()
    solution3 = chinese_remainder_theorem_all_solutions(remainders, moduli)
    time3 = time.time() - start_time
    
    results['all_solutions'] = {
        'solution': solution3,
        'time': time3,
        'success': solution3 is not None
    }
    
    return results


def generate_test_cases() -> List[Tuple[List[int], List[int]]]:
    """Generate various test cases for Chinese Remainder Theorem."""
    return [
        # Simple case
        ([2, 3, 2], [3, 5, 7]),
        
        # Larger numbers
        ([1, 2, 3, 4], [5, 7, 11, 13]),
        
        # Edge case: single congruence
        ([5], [7]),
        
        # Edge case: zero remainders
        ([0, 0, 0], [3, 5, 7]),
        
        # Large numbers
        ([123, 456, 789], [1000000007, 1000000009, 1000000021]),
    ]


def main():
    """Example usage of Chinese Remainder Theorem."""
    print("=== Chinese Remainder Theorem Demo ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, (remainders, moduli) in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Remainders: {remainders}")
        print(f"  Moduli: {moduli}")
        
        # Performance analysis
        metrics = analyze_performance(remainders, moduli)
        
        for method, data in metrics.items():
            print(f"  {method.replace('_', ' ').title()}:")
            if data['success']:
                print(f"    Solution: {data['solution']}")
                print(f"    Time: {data['time']:.6f}s")
                
                # Verify solution
                if isinstance(data['solution'], tuple):
                    solution = data['solution'][0]
                else:
                    solution = data['solution']
                
                is_correct = verify_crt_solution(solution, remainders, moduli)
                print(f"    Correct: {is_correct}")
            else:
                print(f"    No solution exists")
                print(f"    Time: {data['time']:.6f}s")
        
        print()
    
    # Detailed example
    print("=== Detailed Example ===")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    
    print(f"Solving system of congruences:")
    for i, (r, m) in enumerate(zip(remainders, moduli)):
        print(f"  x = {r} (mod {m})")
    
    solution = chinese_remainder_theorem(remainders, moduli)
    if solution is not None:
        print(f"\nSolution: x = {solution}")
        
        # Verify solution
        print(f"\nVerification:")
        for r, m in zip(remainders, moduli):
            print(f"  {solution} = {solution % m} (mod {m})")
        
        # Find all solutions
        all_solutions = chinese_remainder_theorem_all_solutions(remainders, moduli)
        if all_solutions is not None:
            x0, period = all_solutions
            print(f"\nAll solutions: x = {x0} (mod {period})")
            
            # Show first few solutions
            print(f"First few solutions:")
            for i in range(5):
                solution_i = x0 + i * period
                print(f"  x = {solution_i}")
    else:
        print("No solution exists")
    
    print()
    
    # Garner's algorithm example
    print("=== Garner's Algorithm Example ===")
    remainders = [1, 2, 3, 4]
    moduli = [5, 7, 11, 13]  # Pairwise coprime
    
    print(f"Using Garner's algorithm:")
    for i, (r, m) in enumerate(zip(remainders, moduli)):
        print(f"  x = {r} (mod {m})")
    
    solution = chinese_remainder_theorem_garner(remainders, moduli)
    if solution is not None:
        print(f"\nSolution: x = {solution}")
        
        # Verify solution
        print(f"\nVerification:")
        for r, m in zip(remainders, moduli):
            print(f"  {solution} = {solution % m} (mod {m})")
    else:
        print("No solution exists (moduli not pairwise coprime)")


if __name__ == "__main__":
    main() 
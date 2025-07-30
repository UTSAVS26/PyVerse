def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    
    sequence = [0, 1]
    
    while len(sequence) < n:
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    
    return sequence

def find_prime_numbers(max_num):
    """Find all prime numbers up to max_num."""
    primes = []
    
    for num in range(2, max_num + 1):
        is_prime = True
        
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        
        if is_prime:
            primes.append(num)
    
    return primes

def process_data_with_retry(data_list, max_retries=3):
    """Process data with retry logic."""
    results = []
    
    for item in data_list:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Simulate processing
                processed_item = item * 2
                results.append(processed_item)
                success = True
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed to process {item} after {max_retries} retries")
        
        if success:
            print(f"Successfully processed {item}")
    
    return results

def analyze_numbers(numbers):
    """Analyze a list of numbers and return statistics."""
    if not numbers:
        return {}
    
    total = 0
    count = 0
    min_val = float('inf')
    max_val = float('-inf')
    
    for num in numbers:
        total += num
        count += 1
        min_val = min(min_val, num)
        max_val = max(max_val, num)
    
    return {
        'sum': total,
        'count': count,
        'average': total / count if count > 0 else 0,
        'min': min_val,
        'max': max_val
    }

# Example usage
if __name__ == "__main__":
    # Test Fibonacci
    fib_sequence = fibonacci_sequence(10)
    print(f"Fibonacci: {fib_sequence}")
    
    # Test prime numbers
    primes = find_prime_numbers(50)
    print(f"Primes up to 50: {primes}")
    
    # Test data processing
    data = [1, 2, 3, 4, 5]
    processed = process_data_with_retry(data)
    print(f"Processed data: {processed}")
    
    # Test number analysis
    numbers = [10, 20, 30, 40, 50]
    stats = analyze_numbers(numbers)
    print(f"Statistics: {stats}") 
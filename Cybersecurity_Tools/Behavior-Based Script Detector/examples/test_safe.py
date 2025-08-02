#!/usr/bin/env python3
"""
Safe test script for behavior-based script detector.

This script contains only safe operations to test the detector's
ability to correctly identify safe code.
"""

import math
import json
import datetime
import random
import re
from typing import List, Dict, Any

def safe_math_operations():
    """Contains safe mathematical operations."""
    # Basic math operations
    result = math.sqrt(16)
    result = math.pow(2, 3)
    result = math.sin(math.pi / 2)
    
    # Random number generation
    random_number = random.randint(1, 100)
    random_float = random.random()
    
    return result

def safe_string_operations():
    """Contains safe string operations."""
    # String manipulation
    text = "Hello, World!"
    upper_text = text.upper()
    lower_text = text.lower()
    split_text = text.split(',')
    
    # Regular expressions
    pattern = r'\d+'
    matches = re.findall(pattern, "There are 123 numbers and 456 more")
    
    return matches

def safe_data_structures():
    """Contains safe data structure operations."""
    # Lists
    numbers = [1, 2, 3, 4, 5]
    doubled = [x * 2 for x in numbers]
    filtered = [x for x in numbers if x > 2]
    
    # Dictionaries
    person = {
        'name': 'John Doe',
        'age': 30,
        'city': 'New York'
    }
    
    # Sets
    unique_numbers = set([1, 2, 2, 3, 3, 4])
    
    return person

def safe_file_operations():
    """Contains safe file operations."""
    # Safe file reading (only reading, no sensitive paths)
    try:
        with open('data.txt', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = "Default content"
    
    # JSON operations
    data = {'key': 'value', 'number': 42}
    json_string = json.dumps(data)
    parsed_data = json.loads(json_string)
    
    return parsed_data

def safe_datetime_operations():
    """Contains safe datetime operations."""
    # Current time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Date arithmetic
    tomorrow = now + datetime.timedelta(days=1)
    yesterday = now - datetime.timedelta(days=1)
    
    return formatted_time

def safe_function_calls():
    """Contains safe function calls."""
    # Built-in functions
    length = len("Hello")
    maximum = max([1, 2, 3, 4, 5])
    minimum = min([1, 2, 3, 4, 5])
    total = sum([1, 2, 3, 4, 5])
    
    # Type conversions
    number = int("123")
    text = str(456)
    float_num = float("3.14")
    
    return total

def safe_control_structures():
    """Contains safe control structures."""
    # Conditional statements
    x = 10
    if x > 5:
        result = "Greater than 5"
    elif x == 5:
        result = "Equal to 5"
    else:
        result = "Less than 5"
    
    # Loops
    for i in range(5):
        print(f"Loop iteration {i}")
    
    # List comprehension
    squares = [i**2 for i in range(10)]
    
    return result

def safe_error_handling():
    """Contains safe error handling."""
    try:
        # Safe operation that might fail
        result = 10 / 2
    except ZeroDivisionError:
        result = 0
    except Exception as e:
        result = f"Error: {e}"
    finally:
        cleanup = "Cleanup completed"
    
    return result

def main():
    """Main function that calls all safe functions."""
    print("Testing safe behavior patterns...")
    
    # Call all the safe functions
    safe_math_operations()
    safe_string_operations()
    safe_data_structures()
    safe_file_operations()
    safe_datetime_operations()
    safe_function_calls()
    safe_control_structures()
    safe_error_handling()
    
    print("All safe patterns tested!")

if __name__ == "__main__":
    main() 
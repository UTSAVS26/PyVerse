def safe_division(a, b):
    """Perform safe division with exception handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    finally:
        print("Division operation completed.")

def read_file_safely(filename):
    """Read file with comprehensive error handling."""
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except PermissionError:
        print(f"Error: No permission to read '{filename}'.")
        return None
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file '{filename}'.")
        return None
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}")
        return None

def process_data_with_validation(data_list):
    """Process data with validation and error handling."""
    results = []
    
    for i, item in enumerate(data_list):
        try:
            # Validate input
            if not isinstance(item, (int, float)):
                raise ValueError(f"Invalid data type at index {i}")
            
            # Process data
            if item < 0:
                raise ValueError(f"Negative value at index {i}")
            
            processed_item = item * 2
            results.append(processed_item)
            
        except ValueError as ve:
            print(f"Validation error: {ve}")
            results.append(None)
        except Exception as e:
            print(f"Processing error at index {i}: {e}")
            results.append(None)
    
    return results

def connect_to_database(connection_string):
    """Simulate database connection with retry logic."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Simulate connection attempt
            if "invalid" in connection_string:
                raise ConnectionError("Invalid connection string")
            
            print(f"Successfully connected to database (attempt {retry_count + 1})")
            return True
            
        except ConnectionError as ce:
            retry_count += 1
            print(f"Connection failed (attempt {retry_count}): {ce}")
            
            if retry_count >= max_retries:
                print("Max retries reached. Connection failed.")
                return False
            
            print("Retrying...")
    
    return False

def validate_user_input(user_input):
    """Validate user input with multiple checks."""
    errors = []
    
    try:
        # Check if input is not empty
        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty")
        
        # Check length
        if len(user_input) < 3:
            raise ValueError("Input must be at least 3 characters long")
        
        if len(user_input) > 100:
            raise ValueError("Input must be less than 100 characters")
        
        # Check for special characters
        if any(char in user_input for char in ['<', '>', '&', '"', "'"]):
            raise ValueError("Input contains invalid characters")
        
        return True, "Input is valid"
        
    except ValueError as ve:
        return False, str(ve)
    except Exception as e:
        return False, f"Unexpected error: {e}"

# Example usage
if __name__ == "__main__":
    # Test safe division
    print("Testing safe division:")
    print(safe_division(10, 2))
    print(safe_division(10, 0))
    print(safe_division("10", 2))
    
    # Test file reading
    print("\nTesting file reading:")
    content = read_file_safely("nonexistent.txt")
    
    # Test data processing
    print("\nTesting data processing:")
    data = [1, 2, -3, "invalid", 5]
    results = process_data_with_validation(data)
    print(f"Results: {results}")
    
    # Test database connection
    print("\nTesting database connection:")
    connect_to_database("valid_connection")
    connect_to_database("invalid_connection")
    
    # Test user input validation
    print("\nTesting user input validation:")
    test_inputs = ["", "ab", "valid_input", "input<with>special&chars", "a" * 150]
    
    for test_input in test_inputs:
        is_valid, message = validate_user_input(test_input)
        print(f"'{test_input}': {message}") 
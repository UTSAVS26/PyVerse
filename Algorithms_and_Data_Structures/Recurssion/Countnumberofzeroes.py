# Function to count the number of zeroes
def counting_number_of_zeroes(value):
    if len(value) == 0:  # Base case: if the string is empty, return 0
        return 0
    if value[0] == '0':  # If the first character is zero, count it and recurse
        return 1 + counting_number_of_zeroes(value[1:])
    else:
        return counting_number_of_zeroes(value[1:])  # Recurse without counting

# Ask the user to enter the number
number = input("Enter the number where the number of zeroes need to be counted: ")

# Check if the input is a valid number (float or int)
try:
    # Convert to float to validate input
    float_number = float(number)  # Convert the input to a float
    
    # Convert the float to an integer by removing the decimal part (if any)
    sanitized_number = str(int(float_number)).replace('-', '')  # Remove negative sign
    
    # Call the function that counts the zeroes and store the result in 'count'
    count = counting_number_of_zeroes(sanitized_number)

    # Print the final count
    print(f"Number of zeroes: {count}")
except ValueError:
    print("The number entered is not a valid number, try again.")

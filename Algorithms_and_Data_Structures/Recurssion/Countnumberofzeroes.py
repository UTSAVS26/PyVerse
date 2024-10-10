count = 0  # the variable that counts the number of zeroes present

def counting_number_of_zeroes(value):  # function to count the number of zeroes
    global count
    if len(value) == 0:  # checks if the string is empty or not
        return
    if value[0] == '0':  # checks if the value is zero
        count += 1  # increments the zero count
    return counting_number_of_zeroes(value[1:])  # calls itself again

# asks the user to enter the number
number = input("Enter the number where the number of zeroes need to be counted: ")

# Check if the input is a valid number (float or int)
try:
    # Convert to float to validate
    float_number = float(number)  # This works for both floats and integers

    # Convert the number to string and remove decimal point and negative sign
    sanitized_number = str(float_number).replace('.', '').replace('-', '')

    counting_number_of_zeroes(sanitized_number)  # calls the function that counts the number
    print(f"Number of zeroes: {count}")  # prints the count
except ValueError:
    print("The number entered is not a valid number, try again.")  # prints if the input is invalid

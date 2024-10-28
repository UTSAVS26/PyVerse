import re

# A simplified list of common passwords for demonstration purposes
COMMON_PASSWORDS = ['123456', 'password', '123456789', 'qwerty', 'abc123', 'password1']

# Function to evaluate the strength of a password and provide improvement tips
def check_password_strength(password):
    score = 0  # Initialize the score for password strength
    feedback = []  # List to store feedback messages

    # Check the length of the password
    if len(password) >= 12:
        score += 2  # Strong score for long passwords
    elif len(password) >= 8:
        score += 1
    else:
        feedback.append("Your password is too short. It should have at least 8 characters.")

    # Check for lowercase letters
    if re.search("[a-z]", password):
        score += 1
    else:
        feedback.append("Consider adding some lowercase letters to strengthen your password.")

    # Check for uppercase letters
    if re.search("[A-Z]", password):
        score += 1
    else:
        feedback.append("Adding uppercase letters can enhance your password's strength.")

    # Check for digits
    if re.search("[0-9]", password):
        score += 1
    else:
        feedback.append("Don't forget to include numbers to make your password stronger.")

    # Check for special characters
    if re.search("[@#$%^&*!]", password):
        score += 1
    else:
        feedback.append("Including special characters (like @, #, $, etc.) can greatly improve security.")

    # Check if the password is too common
    if password in COMMON_PASSWORDS:
        feedback.append("Warning: This password is quite common. Choose something more unique.")
        score -= 2  # Penalize common passwords

    # Detect repeated characters (e.g., "aaa", "111")
    if re.search(r'(.)\1{2,}', password):
        feedback.append("Try to avoid repeated characters like 'aaa'. They can weaken your password.")
        score -= 1

    # Detect simple sequences like "abcd" or "1234"
    if re.search(r'(?:0123|1234|2345|abcd|qwert)', password.lower()):
        feedback.append("Avoid simple sequences (like '1234' or 'abcd') that are easy to guess.")
        score -= 1

    # Provide overall feedback based on the score
    if score >= 5:
        overall_feedback = "Awesome! Your password is strong."
    elif 3 <= score < 5:
        overall_feedback = "Not bad! Your password is medium-strength. A few improvements could help."
    else:
        overall_feedback = "Oh no! Your password is weak. Please consider changing it for better security."

    return overall_feedback, feedback

# Provide the user with tips for creating a strong password
print("Tips for Creating a Strong Password:")
print("- At least 12 characters long.")
print("- Include a mix of uppercase and lowercase letters.")
print("- Use numbers and special characters (e.g., @, #, $, etc.).")
print("- Avoid common passwords and sequences (like '1234', 'abcd').")
print("- Don't use easily guessable information (like your name or birthday).")

# Prompt the user to input a password in the Jupyter notebook
password = input("\nEnter a password to check its strength: ")
overall_feedback, improvement_tips = check_password_strength(password)

# Display overall feedback and improvement tips in the notebook output
print("\nPassword Strength Feedback:")
print(overall_feedback)  # Print the overall feedback

# Print personalized improvement tips
if improvement_tips:
    print("\nSuggestions to Improve Your Password:")
    for tip in improvement_tips:
        print(f"- {tip}")  # Print each improvement tip
else:
    print("Your password meets all the criteria!")

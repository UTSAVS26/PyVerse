import pywhatkit
import datetime
import random
import re

def validate_phone_number(number):
    pattern = re.compile(r"^\+\d{1,3}\d{10}$")
    return pattern.match(number)

def send_otp(number):
    otp = random.randint(100000, 999999)
    message = f"Dear Customer, your OTP is {otp}. Do not share it with anyone."
    pywhatkit.sendwhatmsg_instantly(number, message)
    print("OTP sent successfully.")

def send_custom_message(number):
    message = input("Enter your custom message: ")
    pywhatkit.sendwhatmsg_instantly(number, message)
    print("Custom message sent successfully.")

def send_scheduled_message(number):
    message = input("Enter your message: ")
    hour = int(input("Enter the hour (24-hour format): "))
    minute = int(input("Enter the minute: "))
    pywhatkit.sendwhatmsg(number, message, hour, minute)
    print("Message scheduled successfully.")

def main():
    # Password check
    correct_password = "password123"
    password = input("Enter the password: ")
    if password != correct_password:
        print("Incorrect password. Access denied.")
        return

    number = input("Enter the phone number (with country code): ")
    if not validate_phone_number(number):
        print("Invalid phone number format. Please enter a valid number.")
        return

    print("""Choose your option: 
    1. Send OTP
    2. Send Custom Message
    3. Send a message at a particular time
    """)
    
    try:
        choice = int(input("Enter your choice: "))
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 3.")
        return

    if choice == 1:
        send_otp(number)
    elif choice == 2:
        send_custom_message(number)
    elif choice == 3:
        send_scheduled_message(number)
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()
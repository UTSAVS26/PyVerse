import os
import re
import streamlit as st
from cryptography.fernet import Fernet
from getpass import getpass

# Load the encryption key
def load_key():
    if not os.path.exists("key.key"):
        st.error("Encryption key not found. Please generate a key first.")
        return None
    return open("key.key", "rb").read()

# Function to generate a new encryption key
def generate_key():
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)
    st.success("Encryption key generated successfully!")

# Encrypt a password and save it to the file
def encrypt_password(website, username, password):
    key = load_key()
    if not key:
        return
    
    fernet = Fernet(key)
    encrypted_password = fernet.encrypt(password.encode())
    
    with open("passwords.txt", "a") as file:
        file.write(f"{website},{username},{encrypted_password.decode()}\n")
    
    st.success(f"Password for {website} saved successfully!")

# Decrypt and display the stored passwords
def view_passwords():
    key = load_key()
    if not key:
        return

    fernet = Fernet(key)
    
    if not os.path.exists("passwords.txt"):
        st.warning("No passwords found.")
        return
    
    with open("passwords.txt", "r") as file:
        for line in file.readlines():
            website, username, encrypted_password = line.strip().split(",")
            decrypted_password = fernet.decrypt(encrypted_password.encode()).decode()
            st.write(f"**Website**: {website} | **Username**: {username} | **Password**: {decrypted_password}")

# Function to check password strength
def check_password_strength(password):
    if len(password) < 8:
        return False, "Password too short. It should be at least 8 characters long."
    if not re.search("[a-z]", password):
        return False, "Password should contain at least one lowercase letter."
    if not re.search("[A-Z]", password):
        return False, "Password should contain at least one uppercase letter."
    if not re.search("[0-9]", password):
        return False, "Password should contain at least one digit."
    if not re.search("[@#$%^&*!]", password):
        return False, "Password should contain at least one special character (e.g., @, #, $, etc.)."
    return True, "Password is strong."

# Streamlit app starts here
st.title("Password Manager")

# Main program
menu = ["Add Password", "View Passwords", "Generate Key", "Exit"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Password":
    st.subheader("Add New Password")

    website = st.text_input("Enter the website name")
    username = st.text_input("Enter the username")
    password = st.text_input("Enter the password", type="password")

    if st.button("Save Password"):
        if website and username and password:
            valid, message = check_password_strength(password)
            if not valid:
                st.warning(f"Password not strong enough: {message}")
            else:
                encrypt_password(website, username, password)
        else:
            st.warning("Please fill in all fields")

elif choice == "View Passwords":
    st.subheader("View Stored Passwords")
    view_passwords()

elif choice == "Generate Key":
    st.subheader("Generate Encryption Key")
    if st.button("Generate Key"):
        generate_key()

elif choice == "Exit":
    st.subheader("Thank you for using the Password Manager!")


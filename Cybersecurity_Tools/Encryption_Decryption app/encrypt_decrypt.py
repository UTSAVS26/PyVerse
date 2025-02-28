import streamlit as st

# Function to encrypt using Caesar Cipher
def caesar_encrypt(plaintext, shift):
    encrypted = ""
    for char in plaintext:
        if char.isalpha():
            shift_amount = shift % 26
            ascii_offset = ord('A') if char.isupper() else ord('a')
            encrypted += chr((ord(char) + shift_amount - ascii_offset) % 26 + ascii_offset)
        else:
            encrypted += char
    return encrypted

# Function to decrypt using Caesar Cipher
def caesar_decrypt(ciphertext, shift):
    return caesar_encrypt(ciphertext, -shift)

# Function to encrypt using Vigenère Cipher
def vignere_encrypt(plaintext, key):
    encrypted = ""
    key_length = len(key)
    for i, char in enumerate(plaintext):
        if char.isalpha():
            shift = ord(key[i % key_length].upper()) - ord('A')
            ascii_offset = ord('A') if char.isupper() else ord('a')
            encrypted += chr((ord(char) + shift - ascii_offset) % 26 + ascii_offset)
        else:
            encrypted += char
    return encrypted

# Function to decrypt using Vigenère Cipher
def vignere_decrypt(ciphertext, key):
    decrypted = ""
    key_length = len(key)
    for i, char in enumerate(ciphertext):
        if char.isalpha():
            shift = ord(key[i % key_length].upper()) - ord('A')
            ascii_offset = ord('A') if char.isupper() else ord('a')
            decrypted += chr((ord(char) - shift - ascii_offset) % 26 + ascii_offset)
        else:
            decrypted += char
    return decrypted

# Streamlit app layout
st.title("Basic Encryption and Decryption App")

message = st.text_area("Enter your message:")
method = st.selectbox("Select encryption/decryption method:", ["Caesar Cipher", "Vigenère Cipher"])
action = st.selectbox("Select action:", ["Encrypt", "Decrypt"])

if method == "Caesar Cipher":
    shift = st.number_input("Enter shift value:", min_value=1, value=3)
    
    if st.button("Submit"):
        if action == "Encrypt":
            result = caesar_encrypt(message, shift)
            st.success(f"Encrypted Message: {result}")
        else:
            result = caesar_decrypt(message, shift)
            st.success(f"Decrypted Message: {result}")

else:  # Vigenère Cipher
    key = st.text_input("Enter key:")
    
    if st.button("Submit"):
        if action == "Encrypt":
            result = vignere_encrypt(message, key)
            st.success(f"Encrypted Message: {result}")
        else:
            result = vignere_decrypt(message, key)
            st.success(f"Decrypted Message: {result}")

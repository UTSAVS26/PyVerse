import streamlit as st
from googletrans import Translator
import streamlit_authenticator as stauth


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
#Defining translation and authentication functions
# Authentication setup
hashed_passwords = stauth.Hasher(['password1', 'password2']).generate()
credentials = {
    "usernames": {
        "user1": {"name": "User One", "password": hashed_passwords[0]},
        "user2": {"name": "User Two", "password": hashed_passwords[1]},
    }
}
authenticator = stauth.Authenticate(
    credentials,
    "encryptionapp",
    "auth_secret_key",  # Replace with a secure key
    cookie_expiry_days=1,
)

name, authentication_status, username = authenticator.login("Login", "main")

if not authentication_status:
    st.error("Please log in to use the app.")
    st.stop()
else:
    st.sidebar.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "sidebar")
    
#implemennting multi-language translation
translator = Translator()

def translate_text(text, src_language, dest_language):
    try:
        translation = translator.translate(text, src=src_language, dest=dest_language)
        return translation.text
    except Exception as e:
        return f"Error in translation: {e}"
#updating streamlit interface for language selection
language_options = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese": "zh-cn",
}

st.sidebar.title("Language Options")
src_language = st.sidebar.selectbox("Input Language:", list(language_options.keys()))
dest_language = st.sidebar.selectbox("Output Language:", list(language_options.keys()))
#adding encryption/decryption with translation
if st.checkbox("Translate Before Processing"):
    message = translate_text(message, language_options[src_language], "en")

if st.button("Submit"):
    if action == "Encrypt":
        result = caesar_encrypt(message, shift) if method == "Caesar Cipher" else vignere_encrypt(message, key)
        if st.checkbox("Translate After Processing"):
            result = translate_text(result, "en", language_options[dest_language])
        st.success(f"Result: {result}")
    else:
        result = caesar_decrypt(message, shift) if method == "Caesar Cipher" else vignere_decrypt(message, key)
        if st.checkbox("Translate After Processing"):
            result = translate_text(result, "en", language_options[dest_language])
        st.success(f"Result: {result}")

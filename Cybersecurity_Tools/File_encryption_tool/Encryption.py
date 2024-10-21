import os
from cryptography.fernet import Fernet
from getpass import getpass

def generate_key(password: str) -> bytes:
    return Fernet.generate_key()

def encrypt_file(file_name: str, password: str):
    key = generate_key(password)
    f = Fernet(key)

    with open(file_name, "rb") as file:
        file_data = file.read()

    encrypted_data = f.encrypt(file_data)

    with open(file_name + ".encrypted", "wb") as file:
        file.write(key + b'\n' + encrypted_data)

    print(f"Encrypted {file_name} as {file_name}.encrypted")

def decrypt_file(encrypted_file_name: str, password: str):
    with open(encrypted_file_name, "rb") as file:
        key = file.readline().strip()
        encrypted_data = file.read()

    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)

    original_file_name = encrypted_file_name.replace(".encrypted", "")
    with open(original_file_name, "wb") as file:
        file.write(decrypted_data)

    print(f"Decrypted {encrypted_file_name} as {original_file_name}")

if __name__ == "__main__":
    action = input("Would you like to (e)ncrypt or (d)ecrypt a file? ").lower()
    
    if action not in ['e', 'd']:
        print("Invalid choice. Please choose 'e' to encrypt or 'd' to decrypt.")
    else:
        file_name = input("Enter the file name: ")
        
        if action == 'e':
            password = getpass("Enter a password for encryption: ")
            encrypt_file(file_name, password)
        elif action == 'd':
            password = getpass("Enter the password for decryption: ")
            decrypt_file(file_name)
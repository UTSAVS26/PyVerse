## **Password Manager**

### 🎯 **Goal**

The goal of the code is to create a simple, secure password manager that encrypts and stores passwords for various websites or services, ensuring they can be safely retrieved later. It checks the strength of passwords before storing them and allows users to manage their passwords using strong encryption techniques.

### 🧾 **Description**

This is a simple password manager built with Streamlit, allowing users to securely store, encrypt, and manage their passwords. The application uses AES encryption (via the `cryptography` library) to securely store passwords (in a passwords.txt file), and it includes functionality to generate encryption keys and check password strength before storing passwords. Users can add passwords, view stored passwords, and generate encryption keys all through an easy-to-use graphical interface.

Key Features
- **Password Encryption**: Encrypts and stores passwords securely.
- **Password Strength Validation**: Ensures the password is strong before storing it.
- **Password Viewing**: Decrypts and displays saved passwords when requested.
- **Key Generation**: Generates and stores a unique encryption key for secure password storage.
- **Cross-Platform**: Runs in a browser, supported across multiple platforms.


### 📚 **Libraries Needed**

The following Python libraries are required to run the project:

- **Streamlit**: For creating the web-based user interface.
- **Cryptography**: To provide AES encryption and decryption.
- **Re**: For regular expression-based password strength checking.
- **Getpass**: For securely handling password inputs (optional for CLI-based password entry).


### 📢 **Conclusion**

This password manager provides a secure and simple way to manage your passwords through encryption. By utilizing Streamlit, it offers a friendly graphical interface, making it easy for users to store and retrieve passwords securely. The application ensures password strength before encryption and allows easy key generation for security.


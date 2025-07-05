# StegaChat - Secure Image Steganography

A secure application for hiding encrypted messages within images using LSB (Least Significant Bit) steganography with AES encryption.

## Features

### 🔐 Security Improvements
- **Proper Password-Based Encryption**: Uses PBKDF2 with 100,000 iterations for key derivation
- **Secure Key Management**: Random salt generation and secure key storage
- **Input Validation**: Comprehensive validation for all inputs
- **Error Handling**: Robust error handling throughout the application

### 🖼️ Steganography Features
- **LSB Encoding**: Hides messages in the least significant bits of image pixels
- **Multi-Format Support**: Works with PNG, JPG, JPEG, BMP, and GIF images
- **Size Validation**: Checks if image is large enough for the message
- **Format Conversion**: Automatically converts images to RGB format

### 🖥️ GUI Features
- **Modern Interface**: Clean, user-friendly Tkinter interface
- **Real-time Status**: Status log showing operation progress
- **File Dialogs**: Easy file selection for input/output
- **Password Input**: Secure password entry for encryption
- **Message Length Check**: Warns if message is too long for selected image

### 🌐 Network Features (Optional)
- **Secure WebSocket Server**: Authentication and encryption support
- **Client Management**: Proper client connection handling
- **Message Broadcasting**: Send messages to multiple clients
- **File Transfer**: Support for file sharing over network

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd StegaChat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

## Usage

### Basic Usage

1. **Launch the application**:
   ```bash
   python app.py
   ```

2. **Encode a message**:
   - Enter your message in the text area
   - Optionally check "Encrypt message" and enter a password
   - Click "Encode & Save"
   - Select an input image
   - Choose where to save the encoded image

3. **Decode a message**:
   - Click "Decode Message"
   - Select the image containing the hidden message
   - If encrypted, enter the password when prompted

### Advanced Features

#### Encryption
- Messages are encrypted using AES-256 with Fernet
- Passwords are derived using PBKDF2 with 100,000 iterations
- Salt is randomly generated and stored with the image
- Key files are saved as `.key` files alongside encoded images

#### Network Communication (Optional)
```python
# Start a secure server
from network.websocket_comm import SecureWebSocketServer
import asyncio

server = SecureWebSocketServer(host="0.0.0.0", port=8765, secret_key="your_secret_key")
start_server = server.start_server()

# Run the server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

```python
# Connect as a client
from network.websocket_comm import SecureWebSocketClient
import asyncio

async def main():
    client = SecureWebSocketClient("ws://localhost:8765", "client1", "your_secret_key")
    if await client.connect():
        await client.send_message("Hello, world!")
        await client.disconnect()

asyncio.run(main())
```

## Security Features

### Cryptographic Security
- **PBKDF2 Key Derivation**: 100,000 iterations for password hashing
- **AES-256 Encryption**: Industry-standard symmetric encryption
- **Random Salt Generation**: Unique salt for each encryption
- **Secure Key Storage**: Keys stored separately from encrypted data

### Input Validation
- **File Existence Checks**: Validates input files exist
- **Image Format Validation**: Ensures images are in supported formats
- **Message Length Validation**: Prevents messages too large for images
- **Password Validation**: Ensures passwords are provided when required

### Error Handling
- **Graceful Failures**: Application continues running after errors
- **User Feedback**: Clear error messages for all failure modes
- **Resource Cleanup**: Proper cleanup of file handles and connections

## File Structure

```
StegaChat/
├── app.py                 # Main application entry point
├── crypto/
│   └── aes.py            # AES encryption/decryption module
├── gui/
│   └── main_ui.py        # Tkinter GUI implementation
├── network/
│   └── websocket_comm.py # Secure WebSocket communication
├── steg/
│   └── image_steg.py     # Image steganography implementation
├── media/
│   └── sample.png        # Sample image for testing
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- **Pillow**: Image processing and manipulation
- **cryptography**: AES encryption and key derivation
- **websockets**: Network communication (optional)

## Security Considerations

### ✅ Implemented Security Measures
- Password-based key derivation using PBKDF2
- Random salt generation for each encryption
- Secure key storage separate from encrypted data
- Input validation and sanitization
- Error handling without information leakage

### ⚠️ Important Notes
- **Password Security**: Choose strong, unique passwords
- **Key File Protection**: Keep `.key` files secure and private
- **Network Security**: Use SSL/TLS in production environments
- **Image Source**: Only use trusted image sources

### 🔒 Best Practices
1. Use strong passwords (12+ characters, mixed case, numbers, symbols)
2. Keep key files secure and don't share them
3. Use different passwords for different messages
4. Regularly update the application and dependencies
5. Use SSL/TLS for network communication in production

## Troubleshooting

### Common Issues

**"Image too small" error**:
- Choose a larger image or shorten your message
- The application shows maximum message length for selected images

**"Decryption failed" error**:
- Ensure you're using the correct password
- Check that the `.key` file is present and not corrupted
- Verify the image hasn't been modified since encoding

**"No valid message found" error**:
- The image may not contain a hidden message
- The image may have been corrupted or modified
- Try a different image file

### Performance Tips
- Use PNG format for best compatibility
- Larger images can hold longer messages
- Network features are optional and can be disabled

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and legitimate use only. Users are responsible for complying with all applicable laws and regulations. The authors are not responsible for any misuse of this software.
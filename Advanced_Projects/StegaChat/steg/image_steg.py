from PIL import Image
import os

def encode_message(img_path: str, message: str, output_path: str) -> bool:
    """
    Encode a message into an image using LSB steganography.
    
    Args:
        img_path: Path to the input image
        message: Message to encode
        output_path: Path to save the encoded image
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If image is too small for the message
        FileNotFoundError: If input image doesn't exist
        OSError: If there are file I/O issues
    """
    try:
        # Validate input file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Input image not found: {img_path}")
        
        # Validate message is not empty
        if not message:
            raise ValueError("Message cannot be empty")
        
        # Open and validate image
        img = Image.open(img_path)
        
        # Convert to RGB if necessary
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        encoded = img.copy()
        width, height = img.size
        
        # Calculate required pixels for the message
        message_bits = len(message) * 8 + 8  # +8 for null terminator
        required_pixels = (message_bits + 2) // 3  # 3 bits per pixel (RGB)
        
        if width * height < required_pixels:
            raise ValueError(
                f"Image too small. Need at least {required_pixels} pixels, "
                f"but image has {width * height} pixels"
            )
        
        # Encode the message
        index = 0
        message += chr(0)  # Null terminator
        
        for row in range(height):
            for col in range(width):
                if index >= len(message) * 8:
                    break
                    
                pixel_data = img.getpixel((col, row))
                # Handle different pixel formats
                if isinstance(pixel_data, int):
                    # Grayscale image
                    pixel = [pixel_data, pixel_data, pixel_data]
                else:
                    # RGB/RGBA image
                    pixel = list(pixel_data)[:3]  # Take only RGB components
                
                for n in range(3):  # R, G, B
                    if index < len(message) * 8:
                        char_index = index // 8
                        bit_index = index % 8
                        bit = (ord(message[char_index]) >> (7 - bit_index)) & 1
                        pixel[n] = pixel[n] & ~1 | bit
                        index += 1
                encoded.putpixel((col, row), tuple(pixel))
            
            if index >= len(message) * 8:
                break
        
        # Save the encoded image
        encoded.save(output_path)
        return True
        
    except (FileNotFoundError, ValueError, OSError):
        raise
    except Exception as e:
        raise OSError(f"Failed to encode message: {e}")
def decode_message(img_path: str) -> str:
    """
    Decode a message from an image using LSB steganography.
    
    Args:
        img_path: Path to the image containing the hidden message
        
    Returns:
        The decoded message
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If no message is found or image is corrupted
    """
    try:
        # Validate input file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Open image
        img = Image.open(img_path)
        
        # Convert to RGB if necessary
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        width, height = img.size
        message = ""
        char = 0
        count = 0
        max_chars = width * height * 3 // 8  # Maximum possible characters
        
        for row in range(height):
            for col in range(width):
                pixel_data = img.getpixel((col, row))
                # Handle different pixel formats
                if isinstance(pixel_data, int):
                    # Grayscale image
                    pixel = [pixel_data, pixel_data, pixel_data]
                else:
                    # RGB/RGBA image
                    pixel = list(pixel_data)[:3]  # Take only RGB components
                
                for n in range(3):  # R, G, B
                    bit = pixel[n] & 1
                    char = (char << 1) | bit
                    count += 1
                    if count == 8:
                        if char == 0:  # Null terminator found
                            return message
                        message += chr(char)
                        char = 0
                        count = 0
                        
                        # Safety check to prevent infinite loops
                        if len(message) > max_chars:
                            raise ValueError("No valid message found or image is corrupted")
        
        raise ValueError("No null terminator found - message may be incomplete or corrupted")
        
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to decode message: {e}")

def get_max_message_length(img_path: str) -> int:
    """
    Calculate the maximum message length that can be encoded in an image.
    
    Args:
        img_path: Path to the image
        
    Returns:
        Maximum number of characters that can be encoded
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        # 3 bits per pixel, 8 bits per character, minus 1 for null terminator
        return (width * height * 3 // 8) - 1
    except Exception as e:
        raise ValueError(f"Could not calculate max message length: {e}")

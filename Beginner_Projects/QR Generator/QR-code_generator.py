import qrcode
from urllib.parse import urlparse
from PIL import ImageColor

# Function to validate the URL
def validate_url(url):
    """Validates the format of the URL."""
    parsed_url = urlparse(url)
    return all([parsed_url.scheme, parsed_url.netloc])

# Function to check if the color input is valid
def validate_color(color):
    """Validates the input color using the PIL library."""
    try:
        ImageColor.getrgb(color)
        return True
    except ValueError:
        return False

# Function to get user input
def get_user_input():
    """Prompts the user for input and validates it."""
    input_URL = input("Enter the URL: ")

    # Validate URL input
    if not validate_url(input_URL):
        raise ValueError("Invalid URL format. Please enter a valid URL.")
    
    # Get and validate color inputs
    fill_color = input("Enter the fill color (default 'black'): ") or "black"
    if not validate_color(fill_color):
        raise ValueError(f"Invalid fill color '{fill_color}'. Please enter a valid color.")
    
    back_color = input("Enter the background color (default 'white'): ") or "white"
    if not validate_color(back_color):
        raise ValueError(f"Invalid background color '{back_color}'. Please enter a valid color.")
    
    # Get box size with basic validation
    try:
        box_size = int(input("Enter the box size (default 15): ") or 15)
        if box_size <= 0:
            raise ValueError("Box size must be a positive integer.")
    except ValueError as ve:
        print("Invalid box size. Using default (15).")
        box_size = 15
    
    return input_URL, fill_color, back_color, box_size

# Function to generate and save the QR code
def generate_qr_code(url, fill_color, back_color, box_size):
    """Generates a QR code with the given parameters and saves it as an image."""
    try:
        # Create QRCode instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Create and save the QR code image
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        img.save("url_qrcode.png")
        print(f"QR code saved as 'url_qrcode.png' for URL: {url}")
        print(qr.data_list)
    except Exception as e:
        raise RuntimeError(f"Error generating the QR code: {e}")

# Main function to run the application
def main():
    """Main entry point to run the QR code generator."""
    try:
        # Get user input and generate QR code
        input_URL, fill_color, back_color, box_size = get_user_input()
        generate_qr_code(input_URL, fill_color, back_color, box_size)
    except ValueError as ve:
        print(f"Input Error: {ve}")
    except RuntimeError as re:
        print(f"Runtime Error: {re}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()


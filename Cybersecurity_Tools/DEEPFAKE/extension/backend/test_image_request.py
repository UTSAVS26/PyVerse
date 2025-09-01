import base64
import requests

# Replace this with your real image path
with open("test_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# For JPG
base64_image = "data:image/jpeg;base64," + encoded_string

# Send request
url = "http://127.0.0.1:8000/detect-image"
response = requests.post(url, json={"image": base64_image})

# Print result
print("✅ Server response:", response.json())

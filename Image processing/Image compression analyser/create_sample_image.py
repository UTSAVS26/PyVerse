#!/usr/bin/env python3
"""
Create a sample test image for the Image Compression Analyzer.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image():
    """Create a sample test image with various features."""
    # Create a 512x512 image
    width, height = 512, 512
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw a gradient background
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            draw.point((x, y), fill=(r, g, b))
    
    # Draw some shapes
    # Circle
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=3)
    
    # Rectangle
    draw.rectangle([200, 50, 350, 150], fill=(0, 255, 0), outline=(0, 0, 0), width=3)
    
    # Triangle
    draw.polygon([(400, 50), (450, 150), (350, 150)], fill=(0, 0, 255), outline=(0, 0, 0), width=3)
    
    # Draw some text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 200), "Sample Image for Compression Analysis", fill=(0, 0, 0), font=font)
    draw.text((50, 230), "Contains gradients, shapes, and text", fill=(0, 0, 0), font=font)
    
    # Draw some lines
    for i in range(10):
        x1 = 50 + i * 40
        y1 = 300
        x2 = x1 + 30
        y2 = y1 + 30
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 0), width=2)
    
    # Add some noise for more realistic testing
    array = np.array(image)
    noise = np.random.normal(0, 10, array.shape).astype(np.uint8)
    noisy_array = np.clip(array + noise, 0, 255)
    image = Image.fromarray(noisy_array)
    
    return image

if __name__ == "__main__":
    # Create sample image
    sample_image = create_sample_image()
    
    # Ensure data directory exists
    os.makedirs("data/input_images", exist_ok=True)
    
    # Save the image
    output_path = "data/input_images/sample_test_image.png"
    sample_image.save(output_path)
    
    print(f"Sample test image created: {output_path}")
    print(f"Image size: {sample_image.size}")
    print(f"Image mode: {sample_image.mode}") 
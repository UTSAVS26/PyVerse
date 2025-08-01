import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to the directory
        extensions: List of file extensions to include (default: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)


def ensure_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_size_kb(file_path: str) -> float:
    """
    Get file size in kilobytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in KB
    """
    return get_file_size(file_path) / 1024


def calculate_size_reduction(original_size: int, compressed_size: int) -> float:
    """
    Calculate size reduction percentage.
    
    Args:
        original_size: Original file size in bytes
        compressed_size: Compressed file size in bytes
    
    Returns:
        Size reduction percentage
    """
    if original_size == 0:
        return 0.0
    return ((original_size - compressed_size) / original_size) * 100


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        PIL Image object
    """
    return Image.open(image_path)


def save_image(image: Image.Image, output_path: str, quality: int = 95) -> None:
    """
    Save an image to file.
    
    Args:
        image: PIL Image object
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))
    
    # Determine format from extension
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg']:
        image.save(output_path, 'JPEG', quality=quality)
    elif ext == '.webp':
        image.save(output_path, 'WEBP', quality=quality)
    elif ext == '.png':
        image.save(output_path, 'PNG')
    elif ext == '.avif':
        image.save(output_path, 'AVIF', quality=quality)
    else:
        image.save(output_path)


def image_to_array(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image object
    
    Returns:
        Numpy array representation of the image
    """
    return np.array(image)


def array_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array
    
    Returns:
        PIL Image object
    """
    return Image.fromarray(array)


def get_image_info(image_path: str) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with image information
    """
    with Image.open(image_path) as img:
        return {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'format': img.format,
            'size_bytes': get_file_size(image_path),
            'size_kb': get_file_size_kb(image_path)
        }


def create_test_image(width: int = 512, height: int = 512, 
                     color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """
    Create a test image for testing purposes.
    
    Args:
        width: Image width
        height: Image height
        color: RGB color tuple
    
    Returns:
        PIL Image object
    """
    array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(array)


def add_noise_to_image(image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
    """
    Add random noise to an image for testing compression algorithms.
    
    Args:
        image: PIL Image object
        noise_factor: Amount of noise to add (0.0 to 1.0)
    
    Returns:
        PIL Image with noise added
    """
    array = image_to_array(image)
    noise = np.random.normal(0, noise_factor * 255, array.shape).astype(np.uint8)
    noisy_array = np.clip(array + noise, 0, 255)
    return Image.fromarray(noisy_array) 
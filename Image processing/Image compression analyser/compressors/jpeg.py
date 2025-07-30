import os
import time
from typing import Tuple, Dict, Any
from PIL import Image
from utils.file_utils import get_file_size, calculate_size_reduction


class JPEGCompressor:
    """JPEG compression implementation."""
    
    def __init__(self):
        self.format_name = "JPEG"
        self.extension = ".jpg"
    
    def compress(self, image: Image.Image, output_path: str, quality: int = 80) -> Dict[str, Any]:
        """
        Compress image using JPEG format.
        
        Args:
            image: PIL Image object
            output_path: Output file path
            quality: JPEG quality (1-100)
        
        Returns:
            Dictionary with compression results
        """
        start_time = time.time()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save with JPEG compression
        image.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        compression_time = time.time() - start_time
        
        # Calculate metrics
        original_size = len(image.tobytes())
        compressed_size = get_file_size(output_path)
        size_reduction = calculate_size_reduction(original_size, compressed_size)
        
        return {
            'format': self.format_name,
            'quality': quality,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'size_reduction_percent': size_reduction,
            'compression_time': compression_time,
            'output_path': output_path
        }
    
    def compress_with_multiple_qualities(self, image: Image.Image, output_dir: str, 
                                       qualities: list = None) -> Dict[str, Any]:
        """
        Compress image with multiple quality settings.
        
        Args:
            image: PIL Image object
            output_dir: Output directory
            qualities: List of quality values to test
        
        Returns:
            Dictionary with results for each quality setting
        """
        if qualities is None:
            qualities = [90, 80, 70, 60, 50]
        
        results = {}
        
        for quality in qualities:
            output_path = os.path.join(output_dir, f"jpeg_q{quality}.jpg")
            result = self.compress(image, output_path, quality)
            results[f"quality_{quality}"] = result
        
        return results
    
    def get_compression_info(self) -> Dict[str, Any]:
        """
        Get information about JPEG compression.
        
        Returns:
            Dictionary with compression format information
        """
        return {
            'format': self.format_name,
            'extension': self.extension,
            'description': 'Joint Photographic Experts Group compression',
            'lossy': True,
            'supports_alpha': False,
            'quality_range': (1, 100),
            'best_for': ['Photographs', 'Web images', 'Digital cameras']
        } 
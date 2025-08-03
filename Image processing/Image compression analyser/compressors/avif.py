import os
import time
from typing import Tuple, Dict, Any
from PIL import Image
from utils.file_utils import get_file_size, calculate_size_reduction


class AVIFCompressor:
    """AVIF compression implementation."""
    
    def __init__(self):
        self.format_name = "AVIF"
        self.extension = ".avif"
    
    def compress(self, image: Image.Image, output_path: str, quality: int = 80) -> Dict[str, Any]:
        """
        Compress image using AVIF format.
        
        Args:
            image: PIL Image object
            output_path: Output file path
            quality: AVIF quality (1-100)
        
        Returns:
            Dictionary with compression results
        """
        start_time = time.time()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Save with AVIF compression
            image.save(output_path, 'AVIF', quality=quality)
        except Exception as e:
            # Fallback to WebP if AVIF is not supported
            print(f"AVIF compression failed: {e}. Falling back to WebP.")
            output_path = output_path.replace('.avif', '.webp')
            image.save(output_path, 'WEBP', quality=quality)
        
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
            output_path = os.path.join(output_dir, f"avif_q{quality}.avif")
            result = self.compress(image, output_path, quality)
            results[f"quality_{quality}"] = result
        
        return results
    
    def compress_lossless(self, image: Image.Image, output_path: str) -> Dict[str, Any]:
        """
        Compress image using lossless AVIF.
        
        Args:
            image: PIL Image object
            output_path: Output file path
        
        Returns:
            Dictionary with compression results
        """
        start_time = time.time()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Save with lossless AVIF compression
            image.save(output_path, 'AVIF', lossless=True)
        except Exception as e:
            # Fallback to lossless WebP if AVIF is not supported
            print(f"Lossless AVIF compression failed: {e}. Falling back to WebP.")
            output_path = output_path.replace('.avif', '.webp')
            image.save(output_path, 'WEBP', lossless=True)
        
        compression_time = time.time() - start_time
        
        # Calculate metrics
        original_size = len(image.tobytes())
        compressed_size = get_file_size(output_path)
        size_reduction = calculate_size_reduction(original_size, compressed_size)
        
        return {
            'format': f"{self.format_name} (Lossless)",
            'quality': 'lossless',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'size_reduction_percent': size_reduction,
            'compression_time': compression_time,
            'output_path': output_path
        }
    
    def get_compression_info(self) -> Dict[str, Any]:
        """
        Get information about AVIF compression.
        
        Returns:
            Dictionary with compression format information
        """
        return {
            'format': self.format_name,
            'extension': self.extension,
            'description': 'AV1 Image File Format',
            'lossy': True,
            'supports_alpha': True,
            'quality_range': (1, 100),
            'best_for': ['Modern web images', 'High-quality photographs', 'Graphics with transparency']
        } 
import imagehash
from PIL import Image
from typing import Dict, Any


class PerceptualHash:
    """Calculate perceptual hashes for image similarity detection."""
    
    def __init__(self):
        self.hash_methods = {
            'average': imagehash.average_hash,
            'phash': imagehash.phash,
            'dhash': imagehash.dhash,
            'whash': imagehash.whash,
            'colorhash': imagehash.colorhash
        }
    
    def calculate_hash(self, image: Image.Image, method: str = 'average') -> str:
        """
        Calculate perceptual hash for an image.
        
        Args:
            image: PIL Image object
            method: Hash method ('average', 'phash', 'dhash', 'whash', 'colorhash')
        
        Returns:
            Hash string
        """
        if method not in self.hash_methods:
            raise ValueError(f"Unknown hash method: {method}")
        
        hash_func = self.hash_methods[method]
        return str(hash_func(image))
    
    def calculate_all_hashes(self, image: Image.Image) -> Dict[str, str]:
        """
        Calculate all types of perceptual hashes for an image.
        
        Args:
            image: PIL Image object
        
        Returns:
            Dictionary with all hash types
        """
        hashes = {}
        for method_name in self.hash_methods.keys():
            hashes[method_name] = self.calculate_hash(image, method_name)
        return hashes
    
    def calculate_hash_difference(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
        
        Returns:
            Hamming distance (number of different bits)
        """
        return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
    
    def compare_images(self, image1: Image.Image, image2: Image.Image, 
                      method: str = 'average') -> Dict[str, Any]:
        """
        Compare two images using perceptual hash.
        
        Args:
            image1: First PIL Image
            image2: Second PIL Image
            method: Hash method to use
        
        Returns:
            Dictionary with comparison results
        """
        hash1 = self.calculate_hash(image1, method)
        hash2 = self.calculate_hash(image2, method)
        difference = self.calculate_hash_difference(hash1, hash2)
        
        # Normalize difference (0 = identical, higher = more different)
        max_difference = 64  # For 64-bit hashes
        similarity_percentage = ((max_difference - difference) / max_difference) * 100
        
        return {
            'hash1': hash1,
            'hash2': hash2,
            'difference': difference,
            'similarity_percentage': similarity_percentage,
            'method': method
        }
    
    def compare_multiple_images(self, original: Image.Image, 
                              compressed_images: Dict[str, Image.Image]) -> Dict[str, Any]:
        """
        Compare original image with multiple compressed versions.
        
        Args:
            original: Original PIL Image
            compressed_images: Dictionary of {name: PIL Image} pairs
        
        Returns:
            Dictionary with comparison results for each compressed image
        """
        results = {}
        
        for name, compressed_img in compressed_images.items():
            comparison = self.compare_images(original, compressed_img)
            results[name] = comparison
        
        return results
    
    def get_similarity_assessment(self, similarity_percentage: float) -> str:
        """
        Get similarity assessment based on percentage.
        
        Args:
            similarity_percentage: Similarity percentage (0-100)
        
        Returns:
            Similarity assessment string
        """
        if similarity_percentage >= 95:
            return "Nearly Identical"
        elif similarity_percentage >= 85:
            return "Very Similar"
        elif similarity_percentage >= 70:
            return "Similar"
        elif similarity_percentage >= 50:
            return "Somewhat Similar"
        else:
            return "Different"
    
    def analyze_compression_impact(self, original_path: str, 
                                 compression_results: dict) -> dict:
        """
        Analyze the impact of compression using perceptual hashing.
        
        Args:
            original_path: Path to original image
            compression_results: Dictionary with compression results
        
        Returns:
            Dictionary with perceptual hash analysis added
        """
        original_image = Image.open(original_path)
        
        for format_name, result in compression_results.items():
            if 'output_path' in result:
                try:
                    compressed_image = Image.open(result['output_path'])
                    
                    # Calculate perceptual hash comparison
                    hash_comparison = self.compare_images(original_image, compressed_image)
                    
                    # Add hash analysis to result
                    result.update({
                        'perceptual_hash': hash_comparison['hash2'],
                        'hash_difference': hash_comparison['difference'],
                        'similarity_percentage': hash_comparison['similarity_percentage'],
                        'similarity_assessment': self.get_similarity_assessment(
                            hash_comparison['similarity_percentage']
                        )
                    })
                except Exception as e:
                    print(f"Error calculating perceptual hash for {format_name}: {e}")
                    result.update({
                        'perceptual_hash': 'Error',
                        'hash_difference': -1,
                        'similarity_percentage': 0.0,
                        'similarity_assessment': 'Error'
                    })
        
        return compression_results 
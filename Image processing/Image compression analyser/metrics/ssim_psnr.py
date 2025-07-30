from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from utils.file_utils import image_to_array


class QualityMetrics:
    """Calculate SSIM and PSNR metrics for image quality assessment."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_ssim(self, original: Image.Image, compressed: Image.Image) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
        
        Returns:
            SSIM value (0-1, higher is better)
        """
        # Convert to grayscale for SSIM calculation
        original_gray = original.convert('L')
        compressed_gray = compressed.convert('L')
        
        # Convert to numpy arrays
        original_array = image_to_array(original_gray)
        compressed_array = image_to_array(compressed_gray)
        
        # Calculate SSIM
        ssim_value = ssim(original_array, compressed_array, data_range=255)
        return ssim_value
    
    def calculate_psnr(self, original: Image.Image, compressed: Image.Image) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
        
        Returns:
            PSNR value in dB (higher is better)
        """
        # Convert to numpy arrays
        original_array = image_to_array(original)
        compressed_array = image_to_array(compressed)
        
        # Calculate PSNR
        psnr_value = psnr(original_array, compressed_array, data_range=255)
        return psnr_value
    
    def calculate_all_metrics(self, original: Image.Image, compressed: Image.Image) -> dict:
        """
        Calculate both SSIM and PSNR metrics.
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
        
        Returns:
            Dictionary with SSIM and PSNR values
        """
        ssim_value = self.calculate_ssim(original, compressed)
        psnr_value = self.calculate_psnr(original, compressed)
        
        return {
            'ssim': ssim_value,
            'psnr': psnr_value,
            'ssim_percentage': ssim_value * 100,
            'psnr_db': psnr_value
        }
    
    def compare_multiple_images(self, original: Image.Image, compressed_images: dict) -> dict:
        """
        Compare original image with multiple compressed versions.
        
        Args:
            original: Original PIL Image
            compressed_images: Dictionary of {name: PIL Image} pairs
        
        Returns:
            Dictionary with metrics for each compressed image
        """
        results = {}
        
        for name, compressed_img in compressed_images.items():
            metrics = self.calculate_all_metrics(original, compressed_img)
            results[name] = metrics
        
        return results
    
    def get_quality_assessment(self, ssim: float, psnr: float) -> str:
        """
        Get quality assessment based on SSIM and PSNR values.
        
        Args:
            ssim: SSIM value (0-1)
            psnr: PSNR value in dB
        
        Returns:
            Quality assessment string
        """
        if ssim >= 0.95 and psnr >= 40:
            return "Excellent"
        elif ssim >= 0.90 and psnr >= 35:
            return "Very Good"
        elif ssim >= 0.80 and psnr >= 30:
            return "Good"
        elif ssim >= 0.70 and psnr >= 25:
            return "Fair"
        else:
            return "Poor"
    
    def calculate_metrics_for_compression_results(self, original_path: str, 
                                                compression_results: dict) -> dict:
        """
        Calculate metrics for compression results.
        
        Args:
            original_path: Path to original image
            compression_results: Dictionary with compression results
        
        Returns:
            Dictionary with metrics added to compression results
        """
        original_image = Image.open(original_path)
        
        for format_name, result in compression_results.items():
            if 'output_path' in result:
                try:
                    compressed_image = Image.open(result['output_path'])
                    metrics = self.calculate_all_metrics(original_image, compressed_image)
                    
                    # Add metrics to result
                    result.update(metrics)
                    result['quality_assessment'] = self.get_quality_assessment(
                        metrics['ssim'], metrics['psnr']
                    )
                except Exception as e:
                    print(f"Error calculating metrics for {format_name}: {e}")
                    result.update({
                        'ssim': 0.0,
                        'psnr': 0.0,
                        'quality_assessment': 'Error'
                    })
        
        return compression_results 
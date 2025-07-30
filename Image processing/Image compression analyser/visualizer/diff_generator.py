import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any
from utils.file_utils import image_to_array, array_to_image


class DiffGenerator:
    """Generate visual differences between original and compressed images."""
    
    def __init__(self):
        self.font_size = 16
        self.text_color = (255, 255, 255)
        self.text_bg_color = (0, 0, 0)
    
    def create_side_by_side_comparison(self, original: Image.Image, compressed: Image.Image,
                                      title: str = "Comparison") -> Image.Image:
        """
        Create a side-by-side comparison of original and compressed images.
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
            title: Title for the comparison
        
        Returns:
            PIL Image with side-by-side comparison
        """
        # Ensure both images are the same size
        width, height = original.size
        compressed = compressed.resize((width, height))
        
        # Create a new image with double width
        comparison = Image.new('RGB', (width * 2, height + 50))
        draw = ImageDraw.Draw(comparison)
        
        # Add title
        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), title, fill=self.text_color, font=font)
        draw.text((10, 30), "Original", fill=self.text_color, font=font)
        draw.text((width + 10, 30), "Compressed", fill=self.text_color, font=font)
        
        # Paste images
        comparison.paste(original, (0, 50))
        comparison.paste(compressed, (width, 50))
        
        return comparison
    
    def create_difference_overlay(self, original: Image.Image, compressed: Image.Image,
                                threshold: int = 30) -> Image.Image:
        """
        Create a difference overlay highlighting areas where images differ.
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
            threshold: Threshold for difference detection
        
        Returns:
            PIL Image with difference overlay
        """
        # Convert to numpy arrays
        original_array = image_to_array(original)
        compressed_array = image_to_array(compressed)
        
        # Ensure same size
        if original_array.shape != compressed_array.shape:
            compressed_array = cv2.resize(compressed_array, 
                                       (original_array.shape[1], original_array.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(original_array, compressed_array)
        
        # Convert to grayscale for thresholding
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        else:
            diff_gray = diff
        
        # Apply threshold
        _, thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Create overlay
        overlay = np.zeros_like(original_array)
        overlay[thresh > 0] = [255, 0, 0]  # Red for differences
        
        # Blend with original
        result = cv2.addWeighted(original_array, 0.7, overlay, 0.3, 0)
        
        return array_to_image(result)
    
    def create_histogram_comparison(self, original: Image.Image, compressed: Image.Image) -> Image.Image:
        """
        Create histogram comparison between original and compressed images.
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
        
        Returns:
            PIL Image with histogram comparison
        """
        # Convert to numpy arrays
        original_array = image_to_array(original)
        compressed_array = image_to_array(compressed)
        
        # Calculate histograms for each channel
        channels = ['Red', 'Green', 'Blue']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Create histogram image
        hist_width = 512
        hist_height = 200
        hist_image = Image.new('RGB', (hist_width, hist_height * 3), (255, 255, 255))
        draw = ImageDraw.Draw(hist_image)
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            # Calculate histograms
            orig_hist = cv2.calcHist([original_array], [i], None, [256], [0, 256])
            comp_hist = cv2.calcHist([compressed_array], [i], None, [256], [0, 256])
            
            # Normalize histograms
            orig_hist = cv2.normalize(orig_hist, orig_hist, 0, hist_height, cv2.NORM_MINMAX)
            comp_hist = cv2.normalize(comp_hist, comp_hist, 0, hist_height, cv2.NORM_MINMAX)
            
            # Draw histograms
            y_offset = i * hist_height
            
            # Draw original histogram (blue)
            for x in range(hist_width):
                x_bin = int(x * 256 / hist_width)
                if x_bin < 256:
                    height = int(orig_hist[x_bin])
                    draw.line([(x, y_offset + hist_height), 
                             (x, y_offset + hist_height - height)], 
                            fill=(0, 0, 255), width=1)
            
            # Draw compressed histogram (red)
            for x in range(hist_width):
                x_bin = int(x * 256 / hist_width)
                if x_bin < 256:
                    height = int(comp_hist[x_bin])
                    draw.line([(x, y_offset + hist_height), 
                             (x, y_offset + hist_height - height)], 
                            fill=(255, 0, 0), width=1)
            
            # Add channel label
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, y_offset + 10), f"{channel} Channel", fill=(0, 0, 0), font=font)
            draw.text((10, y_offset + 25), "Blue=Original, Red=Compressed", fill=(0, 0, 0), font=font)
        
        return hist_image
    
    def create_quality_visualization(self, metrics: Dict[str, Any]) -> Image.Image:
        """
        Create a visual representation of quality metrics.
        
        Args:
            metrics: Dictionary with quality metrics
        
        Returns:
            PIL Image with quality visualization
        """
        width = 400
        height = 300
        
        # Create visualization image
        vis_image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(vis_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        draw.text((10, 10), "Quality Metrics", fill=(0, 0, 0), font=title_font)
        
        y_offset = 50
        
        # Draw SSIM bar
        ssim = metrics.get('ssim', 0)
        ssim_width = int(ssim * 200)
        draw.rectangle([(10, y_offset), (210, y_offset + 20)], outline=(0, 0, 0))
        draw.rectangle([(10, y_offset), (10 + ssim_width, y_offset + 20)], fill=(0, 255, 0))
        draw.text((220, y_offset), f"SSIM: {ssim:.3f}", fill=(0, 0, 0), font=font)
        
        y_offset += 40
        
        # Draw PSNR bar
        psnr = metrics.get('psnr', 0)
        psnr_normalized = min(psnr / 50.0, 1.0)  # Normalize to 0-1
        psnr_width = int(psnr_normalized * 200)
        draw.rectangle([(10, y_offset), (210, y_offset + 20)], outline=(0, 0, 0))
        draw.rectangle([(10, y_offset), (10 + psnr_width, y_offset + 20)], fill=(255, 0, 0))
        draw.text((220, y_offset), f"PSNR: {psnr:.2f} dB", fill=(0, 0, 0), font=font)
        
        y_offset += 40
        
        # Draw size reduction bar
        size_reduction = metrics.get('size_reduction_percent', 0)
        size_width = int(abs(size_reduction) * 2)  # Scale for visualization
        draw.rectangle([(10, y_offset), (210, y_offset + 20)], outline=(0, 0, 0))
        if size_reduction > 0:
            draw.rectangle([(10, y_offset), (10 + size_width, y_offset + 20)], fill=(0, 0, 255))
        draw.text((220, y_offset), f"Size Reduction: {size_reduction:.1f}%", fill=(0, 0, 0), font=font)
        
        return vis_image
    
    def generate_comprehensive_comparison(self, original: Image.Image, compressed: Image.Image,
                                       metrics: Dict[str, Any], output_path: str) -> None:
        """
        Generate a comprehensive comparison image with all visualizations.
        
        Args:
            original: Original PIL Image
            compressed: Compressed PIL Image
            metrics: Quality metrics dictionary
            output_path: Output file path
        """
        # Create side-by-side comparison
        side_by_side = self.create_side_by_side_comparison(original, compressed)
        
        # Create difference overlay
        diff_overlay = self.create_difference_overlay(original, compressed)
        
        # Create histogram comparison
        histogram = self.create_histogram_comparison(original, compressed)
        
        # Create quality visualization
        quality_vis = self.create_quality_visualization(metrics)
        
        # Combine all visualizations
        total_width = max(side_by_side.width, diff_overlay.width, histogram.width, quality_vis.width)
        total_height = side_by_side.height + diff_overlay.height + histogram.height + quality_vis.height + 20
        
        combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # Paste all visualizations
        y_offset = 0
        combined.paste(side_by_side, (0, y_offset))
        y_offset += side_by_side.height + 10
        
        combined.paste(diff_overlay, (0, y_offset))
        y_offset += diff_overlay.height + 10
        
        combined.paste(histogram, (0, y_offset))
        y_offset += histogram.height + 10
        
        combined.paste(quality_vis, (0, y_offset))
        
        # Save combined visualization
        combined.save(output_path) 
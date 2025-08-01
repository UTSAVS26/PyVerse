import unittest
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressors.jpeg import JPEGCompressor
from compressors.webp import WebPCompressor
from compressors.avif import AVIFCompressor
from utils.file_utils import create_test_image, add_noise_to_image


class TestJPEGCompressor(unittest.TestCase):
    """Test cases for JPEG compression."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = JPEGCompressor()
        self.test_image = create_test_image(256, 256, (128, 128, 128))
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.format_name, "JPEG")
        self.assertEqual(self.compressor.extension, ".jpg")
    
    def test_basic_compression(self):
        """Test basic JPEG compression."""
        output_path = os.path.join(self.temp_dir, "test.jpg")
        result = self.compressor.compress(self.test_image, output_path, quality=80)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check result structure
        self.assertIn('format', result)
        self.assertIn('quality', result)
        self.assertIn('compressed_size', result)
        self.assertIn('compression_time', result)
        self.assertEqual(result['format'], "JPEG")
        self.assertEqual(result['quality'], 80)
    
    def test_multiple_qualities(self):
        """Test compression with multiple quality settings."""
        qualities = [90, 80, 70]
        results = self.compressor.compress_with_multiple_qualities(
            self.test_image, self.temp_dir, qualities
        )
        
        self.assertEqual(len(results), len(qualities))
        
        for quality in qualities:
            key = f"quality_{quality}"
            self.assertIn(key, results)
            self.assertEqual(results[key]['quality'], quality)
    
    def test_compression_info(self):
        """Test compression info method."""
        info = self.compressor.get_compression_info()
        
        self.assertEqual(info['format'], "JPEG")
        self.assertEqual(info['extension'], ".jpg")
        self.assertTrue(info['lossy'])
        self.assertFalse(info['supports_alpha'])
        self.assertEqual(info['quality_range'], (1, 100))
    
    def test_compression_with_noise(self):
        """Test compression with noisy image."""
        noisy_image = add_noise_to_image(self.test_image, 0.1)
        output_path = os.path.join(self.temp_dir, "noisy.jpg")
        result = self.compressor.compress(noisy_image, output_path, quality=80)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertIn('compressed_size', result)
    
    def test_different_image_modes(self):
        """Test compression with different image modes."""
        # Test RGB image
        rgb_image = create_test_image(100, 100, (255, 0, 0))
        output_path = os.path.join(self.temp_dir, "rgb.jpg")
        result = self.compressor.compress(rgb_image, output_path, quality=80)
        self.assertTrue(os.path.exists(output_path))
        
        # Test grayscale image
        gray_image = rgb_image.convert('L')
        output_path = os.path.join(self.temp_dir, "gray.jpg")
        result = self.compressor.compress(gray_image, output_path, quality=80)
        self.assertTrue(os.path.exists(output_path))


class TestWebPCompressor(unittest.TestCase):
    """Test cases for WebP compression."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = WebPCompressor()
        self.test_image = create_test_image(256, 256, (128, 128, 128))
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.format_name, "WebP")
        self.assertEqual(self.compressor.extension, ".webp")
    
    def test_basic_compression(self):
        """Test basic WebP compression."""
        output_path = os.path.join(self.temp_dir, "test.webp")
        result = self.compressor.compress(self.test_image, output_path, quality=80)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check result structure
        self.assertIn('format', result)
        self.assertIn('quality', result)
        self.assertIn('compressed_size', result)
        self.assertIn('compression_time', result)
        self.assertEqual(result['format'], "WebP")
        self.assertEqual(result['quality'], 80)
    
    def test_lossless_compression(self):
        """Test lossless WebP compression."""
        output_path = os.path.join(self.temp_dir, "lossless.webp")
        result = self.compressor.compress_lossless(self.test_image, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result['quality'], 'lossless')
        self.assertIn('Lossless', result['format'])
    
    def test_multiple_qualities(self):
        """Test compression with multiple quality settings."""
        qualities = [90, 80, 70]
        results = self.compressor.compress_with_multiple_qualities(
            self.test_image, self.temp_dir, qualities
        )
        
        self.assertEqual(len(results), len(qualities))
        
        for quality in qualities:
            key = f"quality_{quality}"
            self.assertIn(key, results)
            self.assertEqual(results[key]['quality'], quality)
    
    def test_compression_info(self):
        """Test compression info method."""
        info = self.compressor.get_compression_info()
        
        self.assertEqual(info['format'], "WebP")
        self.assertEqual(info['extension'], ".webp")
        self.assertTrue(info['lossy'])
        self.assertTrue(info['supports_alpha'])
        self.assertEqual(info['quality_range'], (1, 100))


class TestAVIFCompressor(unittest.TestCase):
    """Test cases for AVIF compression."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = AVIFCompressor()
        self.test_image = create_test_image(256, 256, (128, 128, 128))
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.format_name, "AVIF")
        self.assertEqual(self.compressor.extension, ".avif")
    
    def test_basic_compression(self):
        """Test basic AVIF compression."""
        output_path = os.path.join(self.temp_dir, "test.avif")
        result = self.compressor.compress(self.test_image, output_path, quality=80)
        
        # Check that file was created (or fallback was used)
        self.assertTrue(os.path.exists(output_path))
        
        # Check result structure
        self.assertIn('format', result)
        self.assertIn('quality', result)
        self.assertIn('compressed_size', result)
        self.assertIn('compression_time', result)
        self.assertEqual(result['quality'], 80)
    
    def test_lossless_compression(self):
        """Test lossless AVIF compression."""
        output_path = os.path.join(self.temp_dir, "lossless.avif")
        result = self.compressor.compress_lossless(self.test_image, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result['quality'], 'lossless')
        self.assertIn('Lossless', result['format'])
    
    def test_multiple_qualities(self):
        """Test compression with multiple quality settings."""
        qualities = [90, 80, 70]
        results = self.compressor.compress_with_multiple_qualities(
            self.test_image, self.temp_dir, qualities
        )
        
        self.assertEqual(len(results), len(qualities))
        
        for quality in qualities:
            key = f"quality_{quality}"
            self.assertIn(key, results)
            self.assertEqual(results[key]['quality'], quality)
    
    def test_compression_info(self):
        """Test compression info method."""
        info = self.compressor.get_compression_info()
        
        self.assertEqual(info['format'], "AVIF")
        self.assertEqual(info['extension'], ".avif")
        self.assertTrue(info['lossy'])
        self.assertTrue(info['supports_alpha'])
        self.assertEqual(info['quality_range'], (1, 100))


class TestCompressionComparison(unittest.TestCase):
    """Test cases for comparing different compression formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.jpeg_compressor = JPEGCompressor()
        self.webp_compressor = WebPCompressor()
        self.avif_compressor = AVIFCompressor()
        self.test_image = create_test_image(512, 512, (128, 128, 128))
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_format_comparison(self):
        """Test comparison between different formats."""
        quality = 80
        results = {}
        
        # Compress with JPEG
        jpeg_path = os.path.join(self.temp_dir, "test.jpg")
        results['jpeg'] = self.jpeg_compressor.compress(self.test_image, jpeg_path, quality)
        
        # Compress with WebP
        webp_path = os.path.join(self.temp_dir, "test.webp")
        results['webp'] = self.webp_compressor.compress(self.test_image, webp_path, quality)
        
        # Compress with AVIF
        avif_path = os.path.join(self.temp_dir, "test.avif")
        results['avif'] = self.avif_compressor.compress(self.test_image, avif_path, quality)
        
        # Check that all files were created
        self.assertTrue(os.path.exists(jpeg_path))
        self.assertTrue(os.path.exists(webp_path))
        self.assertTrue(os.path.exists(avif_path))
        
        # Check that all results have the same quality
        for format_name, result in results.items():
            self.assertEqual(result['quality'], quality)
            self.assertIn('compressed_size', result)
            self.assertIn('compression_time', result)
    
    def test_size_comparison(self):
        """Test that different formats produce different file sizes."""
        quality = 80
        sizes = {}
        
        # Compress with each format
        compressors = {
            'jpeg': self.jpeg_compressor,
            'webp': self.webp_compressor,
            'avif': self.avif_compressor
        }
        
        for format_name, compressor in compressors.items():
            output_path = os.path.join(self.temp_dir, f"test{compressor.extension}")
            result = compressor.compress(self.test_image, output_path, quality)
            sizes[format_name] = result['compressed_size']
        
        # Check that we have different sizes (indicating different compression)
        unique_sizes = set(sizes.values())
        self.assertGreater(len(unique_sizes), 1, "All formats produced the same file size")


if __name__ == '__main__':
    unittest.main() 
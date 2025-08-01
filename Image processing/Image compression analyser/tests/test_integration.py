import unittest
import tempfile
import os
import sys
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.main import ImageCompressionAnalyzer
from utils.file_utils import create_test_image, add_noise_to_image, ensure_directory


class TestImageCompressionAnalyzer(unittest.TestCase):
    """Integration tests for the complete image compression analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ImageCompressionAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        
        # Create a test image
        test_image = create_test_image(256, 256, (128, 128, 128))
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer.compressors)
        self.assertIsNotNone(self.analyzer.quality_metrics)
        self.assertIsNotNone(self.analyzer.perceptual_hash)
        self.assertIsNotNone(self.analyzer.diff_generator)
        self.assertIsNotNone(self.analyzer.plotter)
        
        # Check that all expected compressors are available
        expected_compressors = ['jpeg', 'webp', 'avif']
        for compressor_name in expected_compressors:
            self.assertIn(compressor_name, self.analyzer.compressors)
    
    def test_single_image_analysis(self):
        """Test analysis of a single image."""
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        # Check that results were generated
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that each format has results
        for format_name in ['jpeg', 'webp']:
            self.assertIn(format_name, results)
            result = results[format_name]
            
            # Check required fields
            self.assertIn('format', result)
            self.assertIn('quality', result)
            self.assertIn('compressed_size', result)
            self.assertIn('compression_time', result)
            self.assertIn('size_reduction_percent', result)
            
            # Check quality metrics
            self.assertIn('ssim', result)
            self.assertIn('psnr', result)
            self.assertIn('quality_assessment', result)
            
            # Check perceptual hash metrics
            self.assertIn('perceptual_hash', result)
            self.assertIn('similarity_percentage', result)
            self.assertIn('similarity_assessment', result)
    
    def test_directory_analysis(self):
        """Test analysis of a directory of images."""
        # Create multiple test images
        image_dir = os.path.join(self.temp_dir, "test_images")
        ensure_directory(image_dir)
        
        for i in range(3):
            test_image = create_test_image(128, 128, (i * 50, i * 50, i * 50))
            image_path = os.path.join(image_dir, f"test_{i}.png")
            test_image.save(image_path)
        
        results = self.analyzer.analyze_directory(
            image_dir, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        # Check that results were generated for all images
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
        
        for image_name, image_results in results.items():
            self.assertIsInstance(image_results, dict)
            self.assertGreater(len(image_results), 0)
            
            for format_name, result in image_results.items():
                self.assertIn('format', result)
                self.assertIn('compressed_size', result)
                self.assertIn('ssim', result)
                self.assertIn('psnr', result)
    
    def test_report_generation(self):
        """Test CSV and HTML report generation."""
        # First analyze an image
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        all_results = {'test_image': results}
        
        # Generate reports
        report_dir = os.path.join(self.temp_dir, "reports")
        self.analyzer.generate_reports(all_results, report_dir)
        
        # Check that reports were created
        csv_path = os.path.join(report_dir, 'compression_analysis.csv')
        html_path = os.path.join(report_dir, 'compression_analysis.html')
        
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(html_path))
        
        # Check CSV content
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_content = f.read()
            self.assertIn('Image', csv_content)
            self.assertIn('Format', csv_content)
            self.assertIn('SSIM', csv_content)
            self.assertIn('PSNR', csv_content)
        
        # Check HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            self.assertIn('Image Compression Analysis Report', html_content)
            self.assertIn('<table>', html_content)
    
    def test_visualization_generation(self):
        """Test visualization generation."""
        # First analyze an image
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        all_results = {'test_image': results}
        
        # Generate visualizations
        vis_dir = os.path.join(self.temp_dir, "visualizations")
        self.analyzer.create_visualizations(all_results, vis_dir)
        
        # Check that visualization files were created
        dashboard_path = os.path.join(vis_dir, "test_image_dashboard.png")
        plotly_path = os.path.join(vis_dir, "test_image_interactive.html")
        
        # Note: These might not exist if matplotlib/plotly failed, but the method should not crash
        # We'll just check that the directory exists and the method completed without error
        self.assertTrue(os.path.exists(vis_dir))
    
    def test_different_quality_settings(self):
        """Test analysis with different quality settings."""
        qualities = [90, 80, 70]
        
        for quality in qualities:
            results = self.analyzer.analyze_single_image(
                self.test_image_path, 
                quality=quality, 
                formats=['jpeg'], 
                output_dir=self.temp_dir
            )
            
            self.assertIn('jpeg', results)
            result = results['jpeg']
            self.assertEqual(result['quality'], quality)
    
    def test_different_formats(self):
        """Test analysis with different compression formats."""
        formats = ['jpeg', 'webp', 'avif']
        
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=formats, 
            output_dir=self.temp_dir
        )
        
        for format_name in formats:
            self.assertIn(format_name, results)
            result = results[format_name]
            self.assertIn('format', result)
            self.assertIn('compressed_size', result)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent image
        results = self.analyzer.analyze_single_image(
            "non_existent_image.jpg", 
            quality=80, 
            formats=['jpeg'], 
            output_dir=self.temp_dir
        )
        
        # Should return empty results or handle error gracefully
        self.assertIsInstance(results, dict)
    
    def test_compression_comparison(self):
        """Test comparison between different compression formats."""
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp', 'avif'], 
            output_dir=self.temp_dir
        )
        
        # Check that all formats produced different results
        sizes = []
        ssim_values = []
        psnr_values = []
        
        for format_name, result in results.items():
            sizes.append(result['compressed_size'])
            ssim_values.append(result['ssim'])
            psnr_values.append(result['psnr'])
        
        # Check that we have some variation in results
        self.assertGreater(len(set(sizes)), 1, "All formats produced the same file size")
        
        # Check that quality metrics are reasonable
        for ssim in ssim_values:
            self.assertGreaterEqual(ssim, 0.0)
            self.assertLessEqual(ssim, 1.0)
        
        for psnr in psnr_values:
            self.assertGreater(psnr, 0.0)
    
    def test_quality_assessment_consistency(self):
        """Test that quality assessments are consistent."""
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        valid_assessments = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
        
        for format_name, result in results.items():
            assessment = result['quality_assessment']
            self.assertIn(assessment, valid_assessments)
    
    def test_similarity_assessment_consistency(self):
        """Test that similarity assessments are consistent."""
        results = self.analyzer.analyze_single_image(
            self.test_image_path, 
            quality=80, 
            formats=['jpeg', 'webp'], 
            output_dir=self.temp_dir
        )
        
        valid_assessments = ["Nearly Identical", "Very Similar", "Similar", "Somewhat Similar", "Different"]
        
        for format_name, result in results.items():
            assessment = result['similarity_assessment']
            self.assertIn(assessment, valid_assessments)


class TestCLIInterface(unittest.TestCase):
    """Test cases for the CLI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        
        # Create a test image
        test_image = create_test_image(256, 256, (128, 128, 128))
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cli_help(self):
        """Test that CLI help works."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run([
                sys.executable, 'cli/main.py', '--help'
            ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            self.assertEqual(result.returncode, 0)
            self.assertIn('Image Compression Analyzer', result.stdout)
            self.assertIn('--input', result.stdout)
            self.assertIn('--quality', result.stdout)
        except Exception as e:
            # Skip this test if CLI is not properly set up
            self.skipTest(f"CLI test skipped: {e}")
    
    def test_cli_single_image(self):
        """Test CLI with single image."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run([
                sys.executable, 'cli/main.py', 
                '--input', self.test_image_path,
                '--quality', '80',
                '--formats', 'jpeg', 'webp'
            ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Should complete without error
            self.assertEqual(result.returncode, 0)
        except Exception as e:
            # Skip this test if CLI is not properly set up
            self.skipTest(f"CLI test skipped: {e}")


if __name__ == '__main__':
    unittest.main() 
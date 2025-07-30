#!/usr/bin/env python3
"""
Image Compression Analyzer - Main CLI Interface

A comprehensive tool for analyzing and comparing image compression techniques.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressors.jpeg import JPEGCompressor
from compressors.webp import WebPCompressor
from compressors.avif import AVIFCompressor
from metrics.ssim_psnr import QualityMetrics
from metrics.perceptual_hash import PerceptualHash
from visualizer.diff_generator import DiffGenerator
from visualizer.plot_metrics import MetricsPlotter
from utils.file_utils import get_image_files, ensure_directory, load_image, create_test_image


class ImageCompressionAnalyzer:
    """Main analyzer class that orchestrates the compression analysis."""
    
    def __init__(self):
        self.compressors = {
            'jpeg': JPEGCompressor(),
            'webp': WebPCompressor(),
            'avif': AVIFCompressor()
        }
        self.quality_metrics = QualityMetrics()
        self.perceptual_hash = PerceptualHash()
        self.diff_generator = DiffGenerator()
        self.plotter = MetricsPlotter()
    
    def analyze_single_image(self, image_path: str, quality: int = 80, 
                           formats: List[str] = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze compression for a single image.
        
        Args:
            image_path: Path to the input image
            quality: Compression quality (1-100)
            formats: List of formats to test
            output_dir: Output directory for results
        
        Returns:
            Dictionary with analysis results
        """
        if formats is None:
            formats = ['jpeg', 'webp', 'avif']
        
        if output_dir is None:
            output_dir = 'data/results'
        
        ensure_directory(output_dir)
        
        # Load image
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {}
        
        results = {}
        
        # Compress with each format
        for format_name in formats:
            if format_name in self.compressors:
                compressor = self.compressors[format_name]
                output_path = os.path.join(output_dir, f"{Path(image_path).stem}_{format_name}.{compressor.extension}")
                
                try:
                    result = compressor.compress(image, output_path, quality)
                    results[format_name] = result
                except Exception as e:
                    print(f"Error compressing with {format_name}: {e}")
                    results[format_name] = {'error': str(e)}
        
        # Calculate quality metrics
        if results:
            results = self.quality_metrics.calculate_metrics_for_compression_results(image_path, results)
            results = self.perceptual_hash.analyze_compression_impact(image_path, results)
        
        return results
    
    def analyze_directory(self, input_dir: str, quality: int = 80, 
                        formats: List[str] = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze compression for all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            quality: Compression quality (1-100)
            formats: List of formats to test
            output_dir: Output directory for results
        
        Returns:
            Dictionary with analysis results for all images
        """
        if formats is None:
            formats = ['jpeg', 'webp', 'avif']
        
        if output_dir is None:
            output_dir = 'data/results'
        
        ensure_directory(output_dir)
        
        # Get all image files
        image_files = get_image_files(input_dir)
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return {}
        
        all_results = {}
        
        for image_path in image_files:
            print(f"Analyzing {image_path}...")
            image_name = Path(image_path).stem
            image_output_dir = os.path.join(output_dir, image_name)
            
            results = self.analyze_single_image(image_path, quality, formats, image_output_dir)
            all_results[image_name] = results
        
        return all_results
    
    def generate_reports(self, results: Dict[str, Any], output_dir: str = None) -> None:
        """
        Generate CSV and HTML reports from analysis results.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory for reports
        """
        if output_dir is None:
            output_dir = 'reports'
        
        ensure_directory(output_dir)
        
        # Generate CSV report
        csv_path = os.path.join(output_dir, 'compression_analysis.csv')
        self._generate_csv_report(results, csv_path)
        
        # Generate HTML report
        html_path = os.path.join(output_dir, 'compression_analysis.html')
        self._generate_html_report(results, html_path)
        
        print(f"Reports generated: {csv_path}, {html_path}")
    
    def _generate_csv_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate CSV report from results."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Image', 'Format', 'Quality', 'SSIM', 'PSNR', 'Size (KB)', 
                         'Size Reduction (%)', 'Compression Time (ms)', 'Quality Assessment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for image_name, image_results in results.items():
                for format_name, result in image_results.items():
                    if isinstance(result, dict) and 'format' in result:
                        row = {
                            'Image': image_name,
                            'Format': result.get('format', format_name),
                            'Quality': result.get('quality', 'N/A'),
                            'SSIM': f"{result.get('ssim', 0):.3f}",
                            'PSNR': f"{result.get('psnr', 0):.2f}",
                            'Size (KB)': f"{result.get('compressed_size', 0) / 1024:.1f}",
                            'Size Reduction (%)': f"{result.get('size_reduction_percent', 0):.1f}",
                            'Compression Time (ms)': f"{result.get('compression_time', 0) * 1000:.1f}",
                            'Quality Assessment': result.get('quality_assessment', 'N/A')
                        }
                        writer.writerow(row)
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate HTML report from results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Compression Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .header { background-color: #4CAF50; color: white; padding: 15px; }
                .summary { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Image Compression Analysis Report</h1>
            </div>
            <div class="summary">
                <h2>Summary</h2>
                <p>This report contains the results of image compression analysis comparing different formats.</p>
            </div>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Format</th>
                    <th>Quality</th>
                    <th>SSIM</th>
                    <th>PSNR (dB)</th>
                    <th>Size (KB)</th>
                    <th>Size Reduction (%)</th>
                    <th>Compression Time (ms)</th>
                    <th>Quality Assessment</th>
                </tr>
        """
        
        for image_name, image_results in results.items():
            for format_name, result in image_results.items():
                if isinstance(result, dict) and 'format' in result:
                    html_content += f"""
                <tr>
                    <td>{image_name}</td>
                    <td>{result.get('format', format_name)}</td>
                    <td>{result.get('quality', 'N/A')}</td>
                    <td>{result.get('ssim', 0):.3f}</td>
                    <td>{result.get('psnr', 0):.2f}</td>
                    <td>{result.get('compressed_size', 0) / 1024:.1f}</td>
                    <td>{result.get('size_reduction_percent', 0):.1f}</td>
                    <td>{result.get('compression_time', 0) * 1000:.1f}</td>
                    <td>{result.get('quality_assessment', 'N/A')}</td>
                </tr>
                    """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = None) -> None:
        """
        Create visualizations for the analysis results.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory for visualizations
        """
        if output_dir is None:
            output_dir = 'data/results'
        
        ensure_directory(output_dir)
        
        # Create charts for each image
        for image_name, image_results in results.items():
            if image_results:
                # Create comprehensive dashboard
                dashboard_path = os.path.join(output_dir, f"{image_name}_dashboard.png")
                self.plotter.create_comprehensive_dashboard(image_results, dashboard_path)
                
                # Create interactive Plotly dashboard
                plotly_path = os.path.join(output_dir, f"{image_name}_interactive.html")
                self.plotter.create_interactive_plotly_dashboard(image_results, plotly_path)
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Image Compression Analyzer - Compare different compression formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/main.py --input data/input_images --quality 80
  python cli/main.py --input sample.jpg --formats jpeg webp avif
  python cli/main.py --input data/input_images --report --output reports/
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input image file or directory')
    parser.add_argument('--quality', '-q', type=int, default=80,
                       help='Compression quality (1-100, default: 80)')
    parser.add_argument('--formats', '-f', nargs='+', 
                       choices=['jpeg', 'webp', 'avif'],
                       default=['jpeg', 'webp', 'avif'],
                       help='Compression formats to test')
    parser.add_argument('--output', '-o', default='data/results',
                       help='Output directory for results')
    parser.add_argument('--report', action='store_true',
                       help='Generate CSV and HTML reports')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations and charts')
    parser.add_argument('--compare', action='store_true',
                       help='Create side-by-side comparisons')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ImageCompressionAnalyzer()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        print(f"Analyzing single image: {args.input}")
        results = analyzer.analyze_single_image(args.input, args.quality, args.formats, args.output)
        all_results = {Path(args.input).stem: results}
    elif os.path.isdir(args.input):
        print(f"Analyzing directory: {args.input}")
        all_results = analyzer.analyze_directory(args.input, args.quality, args.formats, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    # Generate reports if requested
    if args.report:
        analyzer.generate_reports(all_results, args.output)
    
    # Create visualizations if requested
    if args.visualize:
        analyzer.create_visualizations(all_results, args.output)
    
    # Print summary
    print("\nAnalysis Summary:")
    for image_name, image_results in all_results.items():
        print(f"\n{image_name}:")
        for format_name, result in image_results.items():
            if isinstance(result, dict) and 'format' in result:
                size_kb = result.get('compressed_size', 0) / 1024
                ssim = result.get('ssim', 0)
                psnr = result.get('psnr', 0)
                reduction = result.get('size_reduction_percent', 0)
                print(f"  {format_name}: {size_kb:.1f} KB, SSIM: {ssim:.3f}, PSNR: {psnr:.2f} dB, Reduction: {reduction:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
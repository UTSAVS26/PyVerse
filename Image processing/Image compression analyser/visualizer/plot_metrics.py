import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class MetricsPlotter:
    """Generate charts and plots for compression analysis results."""
    
    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use(self.style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_size_comparison_chart(self, results: Dict[str, Any], 
                                   output_path: str = None) -> plt.Figure:
        """
        Create a bar chart comparing file sizes.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Matplotlib figure
        """
        formats = []
        sizes = []
        colors = []
        
        for format_name, result in results.items():
            if 'compressed_size' in result:
                formats.append(format_name)
                sizes.append(result['compressed_size'] / 1024)  # Convert to KB
                colors.append(self.colors[len(formats) % len(self.colors)])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(formats, sizes, color=colors)
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{size:.1f} KB', ha='center', va='bottom')
        
        ax.set_title('File Size Comparison')
        ax.set_ylabel('Size (KB)')
        ax.set_xlabel('Compression Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_quality_comparison_chart(self, results: Dict[str, Any], 
                                      output_path: str = None) -> plt.Figure:
        """
        Create a scatter plot comparing SSIM vs PSNR.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Matplotlib figure
        """
        formats = []
        ssim_values = []
        psnr_values = []
        colors = []
        
        for format_name, result in results.items():
            if 'ssim' in result and 'psnr' in result:
                formats.append(format_name)
                ssim_values.append(result['ssim'])
                psnr_values.append(result['psnr'])
                colors.append(self.colors[len(formats) % len(self.colors)])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(ssim_values, psnr_values, c=colors, s=100, alpha=0.7)
        
        # Add format labels
        for i, format_name in enumerate(formats):
            ax.annotate(format_name, (ssim_values[i], psnr_values[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('SSIM')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Quality Comparison: SSIM vs PSNR')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_size_reduction_chart(self, results: Dict[str, Any], 
                                  output_path: str = None) -> plt.Figure:
        """
        Create a bar chart showing size reduction percentages.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Matplotlib figure
        """
        formats = []
        reductions = []
        colors = []
        
        for format_name, result in results.items():
            if 'size_reduction_percent' in result:
                formats.append(format_name)
                reductions.append(result['size_reduction_percent'])
                colors.append(self.colors[len(formats) % len(self.colors)])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(formats, reductions, color=colors)
        
        # Add value labels on bars
        for bar, reduction in zip(bars, reductions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{reduction:.1f}%', ha='center', va='bottom')
        
        ax.set_title('Size Reduction Comparison')
        ax.set_ylabel('Size Reduction (%)')
        ax.set_xlabel('Compression Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_compression_time_chart(self, results: Dict[str, Any], 
                                    output_path: str = None) -> plt.Figure:
        """
        Create a bar chart showing compression times.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Matplotlib figure
        """
        formats = []
        times = []
        colors = []
        
        for format_name, result in results.items():
            if 'compression_time' in result:
                formats.append(format_name)
                times.append(result['compression_time'] * 1000)  # Convert to ms
                colors.append(self.colors[len(formats) % len(self.colors)])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(formats, times, color=colors)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{time_val:.1f} ms', ha='center', va='bottom')
        
        ax.set_title('Compression Time Comparison')
        ax.set_ylabel('Time (ms)')
        ax.set_xlabel('Compression Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_dashboard(self, results: Dict[str, Any], 
                                    output_path: str = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        formats = []
        sizes = []
        ssim_values = []
        psnr_values = []
        reductions = []
        times = []
        
        for format_name, result in results.items():
            if all(key in result for key in ['compressed_size', 'ssim', 'psnr', 
                                           'size_reduction_percent', 'compression_time']):
                formats.append(format_name)
                sizes.append(result['compressed_size'] / 1024)
                ssim_values.append(result['ssim'])
                psnr_values.append(result['psnr'])
                reductions.append(result['size_reduction_percent'])
                times.append(result['compression_time'] * 1000)
        
        colors = self.colors[:len(formats)]
        
        # Chart 1: File Size
        ax1.bar(formats, sizes, color=colors)
        ax1.set_title('File Size Comparison')
        ax1.set_ylabel('Size (KB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Chart 2: Quality (SSIM vs PSNR)
        scatter = ax2.scatter(ssim_values, psnr_values, c=colors, s=100, alpha=0.7)
        for i, format_name in enumerate(formats):
            ax2.annotate(format_name, (ssim_values[i], psnr_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('SSIM')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Quality Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Size Reduction
        ax3.bar(formats, reductions, color=colors)
        ax3.set_title('Size Reduction')
        ax3.set_ylabel('Reduction (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Chart 4: Compression Time
        ax4.bar(formats, times, color=colors)
        ax4.set_title('Compression Time')
        ax4.set_ylabel('Time (ms)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_plotly_dashboard(self, results: Dict[str, Any], 
                                          output_path: str = None) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results: Dictionary with compression results
            output_path: Optional output path for saving
        
        Returns:
            Plotly figure
        """
        # Prepare data
        formats = []
        sizes = []
        ssim_values = []
        psnr_values = []
        reductions = []
        times = []
        
        for format_name, result in results.items():
            if all(key in result for key in ['compressed_size', 'ssim', 'psnr', 
                                           'size_reduction_percent', 'compression_time']):
                formats.append(format_name)
                sizes.append(result['compressed_size'] / 1024)
                ssim_values.append(result['ssim'])
                psnr_values.append(result['psnr'])
                reductions.append(result['size_reduction_percent'])
                times.append(result['compression_time'] * 1000)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('File Size Comparison', 'Quality Comparison', 
                          'Size Reduction', 'Compression Time'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=formats, y=sizes, name='File Size (KB)', marker_color=self.colors),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ssim_values, y=psnr_values, mode='markers+text',
                      text=formats, textposition="top center", name='Quality',
                      marker=dict(size=15, color=self.colors)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=formats, y=reductions, name='Size Reduction (%)', marker_color=self.colors),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=formats, y=times, name='Compression Time (ms)', marker_color=self.colors),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Image Compression Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Format", row=1, col=1)
        fig.update_yaxes(title_text="Size (KB)", row=1, col=1)
        fig.update_xaxes(title_text="SSIM", row=1, col=2)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="Format", row=2, col=1)
        fig.update_yaxes(title_text="Reduction (%)", row=2, col=1)
        fig.update_xaxes(title_text="Format", row=2, col=2)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=2)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig 
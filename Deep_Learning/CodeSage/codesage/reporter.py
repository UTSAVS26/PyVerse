"""
Report Generator Module

Generates comprehensive reports with AI-enhanced insights and visualizations.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .analyzer import AnalysisResult
from .metrics import FileMetrics, FunctionMetrics


class ReportGenerator:
    """AI-enhanced report generator with rich visualizations."""
    
    def __init__(self):
        self.console = Console()
        
    def generate_cli_report(self, result: AnalysisResult, detailed: bool = False) -> None:
        """Generate a rich CLI report with AI insights."""
        self.console.print("\n" + "="*80)
        self.console.print("[bold blue]🧩 CodeSage Analysis Report[/bold blue]")
        self.console.print("="*80)
        
        # Project Overview
        self._print_project_overview(result.project_metrics)
        
        # AI Insights
        if result.ai_insights:
            self._print_ai_insights(result.ai_insights)
        
        # Risk Hotspots
        if result.risk_hotspots:
            self._print_risk_hotspots(result.risk_hotspots)
        
        # File Analysis
        if detailed:
            self._print_detailed_file_analysis(result.files)
        else:
            self._print_summary_file_analysis(result.files)
        
        # Suggestions
        if result.suggestions:
            self._print_suggestions(result.suggestions)
        
        # Summary
        self._print_summary(result)
    
    def generate_html_report(self, result: AnalysisResult, output_path: str = "codesage_report.html") -> str:
        """Generate an interactive HTML report with AI-enhanced visualizations."""
        html_content = self._generate_html_content(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _print_project_overview(self, project_metrics: Dict) -> None:
        """Print project overview with AI-enhanced metrics."""
        self.console.print("\n[bold cyan]📊 Project Overview[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Basic metrics
        table.add_row("Total Files", str(project_metrics.get('total_files', 0)), "📁")
        table.add_row("Total Lines", str(project_metrics.get('total_lines', 0)), "📝")
        table.add_row("Total Functions", str(project_metrics.get('total_functions', 0)), "⚙️")
        
        # AI-enhanced metrics
        avg_complexity = project_metrics.get('average_complexity', 0)
        complexity_status = self._get_complexity_status(avg_complexity)
        table.add_row("Avg Complexity", f"{avg_complexity:.1f}", complexity_status)
        
        avg_maintainability = project_metrics.get('average_maintainability', 100)
        maintainability_status = self._get_maintainability_status(avg_maintainability)
        table.add_row("Avg Maintainability", f"{avg_maintainability:.1f}/100", maintainability_status)
        
        risk_level = project_metrics.get('risk_level', 'UNKNOWN')
        risk_status = self._get_risk_status(risk_level)
        table.add_row("AI Risk Level", risk_level, risk_status)
        
        self.console.print(table)
        
        # Complexity distribution
        if 'complexity_distribution' in project_metrics:
            self._print_complexity_distribution(project_metrics['complexity_distribution'])
    
    def _print_ai_insights(self, ai_insights: List[str]) -> None:
        """Print AI-generated insights."""
        self.console.print("\n[bold green]🤖 AI Insights[/bold green]")
        
        for i, insight in enumerate(ai_insights, 1):
            self.console.print(f"  {i}. {insight}")
    
    def _print_risk_hotspots(self, risk_hotspots: List[Dict]) -> None:
        """Print identified risk hotspots."""
        self.console.print("\n[bold red]🔥 Risk Hotspots[/bold red]")
        
        table = Table(show_header=True, header_style="bold red")
        table.add_column("File", style="cyan")
        table.add_column("Function", style="yellow")
        table.add_column("Risk Score", style="red")
        table.add_column("Complexity", style="green")
        table.add_column("Risk Type", style="magenta")
        
        for hotspot in risk_hotspots[:5]:  # Show top 5
            risk_score = hotspot['risk_score']
            risk_color = "red" if risk_score > 80 else "yellow" if risk_score > 60 else "green"
            
            table.add_row(
                hotspot['file'],
                hotspot['function'],
                f"{risk_score:.1f}",
                str(hotspot['complexity']),
                hotspot['risk_type']
            )
        
        self.console.print(table)
    
    def _print_detailed_file_analysis(self, files: List[FileMetrics]) -> None:
        """Print detailed analysis for each file."""
        self.console.print("\n[bold blue]📁 Detailed File Analysis[/bold blue]")
        
        for file_metric in files:
            self.console.print(f"\n[cyan]{file_metric.filename}[/cyan]")
            
            # File-level metrics
            file_table = Table(show_header=True, header_style="bold")
            file_table.add_column("Metric", style="cyan")
            file_table.add_column("Value", style="green")
            
            file_table.add_row("Total Lines", str(file_metric.total_lines))
            file_table.add_row("Functions", str(file_metric.total_functions))
            file_table.add_row("Avg Complexity", f"{file_metric.average_complexity:.1f}")
            file_table.add_row("Maintainability", f"{file_metric.maintainability_index:.1f}/100")
            file_table.add_row("AI Anomaly Score", f"{file_metric.ai_anomaly_score:.1f}")
            
            self.console.print(file_table)
            
            # Function-level details
            if file_metric.functions:
                func_table = Table(show_header=True, header_style="bold")
                func_table.add_column("Function", style="yellow")
                func_table.add_column("Complexity", style="green")
                func_table.add_column("Lines", style="blue")
                func_table.add_column("Nesting", style="magenta")
                func_table.add_column("Risk Score", style="red")
                
                for func in file_metric.functions:
                    risk_color = "red" if func.ai_risk_score > 70 else "yellow" if func.ai_risk_score > 40 else "green"
                    func_table.add_row(
                        func.name,
                        str(func.cyclomatic_complexity),
                        str(func.lines_of_code),
                        str(func.nesting_depth),
                        f"[{risk_color}]{func.ai_risk_score:.1f}[/{risk_color}]"
                    )
                
                self.console.print(func_table)
    
    def _print_summary_file_analysis(self, files: List[FileMetrics]) -> None:
        """Print summary file analysis."""
        self.console.print("\n[bold blue]📁 File Summary[/bold blue]")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Functions", style="green")
        table.add_column("Avg Complexity", style="yellow")
        table.add_column("Maintainability", style="blue")
        table.add_column("AI Risk", style="red")
        
        for file_metric in files:
            risk_color = "red" if file_metric.ai_anomaly_score > 70 else "yellow" if file_metric.ai_anomaly_score > 40 else "green"
            
            table.add_row(
                file_metric.filename,
                str(file_metric.total_functions),
                f"{file_metric.average_complexity:.1f}",
                f"{file_metric.maintainability_index:.1f}",
                f"[{risk_color}]{file_metric.ai_anomaly_score:.1f}[/{risk_color}]"
            )
        
        self.console.print(table)
    
    def _print_suggestions(self, suggestions: List[str]) -> None:
        """Print AI-generated suggestions."""
        self.console.print("\n[bold yellow]💡 Suggestions for Improvement[/bold yellow]")
        
        for i, suggestion in enumerate(suggestions, 1):
            self.console.print(f"  {i}. {suggestion}")
    
    def _print_summary(self, result: AnalysisResult) -> None:
        """Print analysis summary."""
        self.console.print("\n[bold green]📋 Summary[/bold green]")
        
        total_suggestions = len(result.suggestions)
        total_hotspots = len(result.risk_hotspots)
        total_insights = len(result.ai_insights)
        
        summary_text = f"""
        • Analyzed {result.project_metrics.get('total_files', 0)} files with {result.project_metrics.get('total_functions', 0)} functions
        • Generated {total_insights} AI insights
        • Identified {total_hotspots} risk hotspots
        • Provided {total_suggestions} improvement suggestions
        • Overall risk level: {result.project_metrics.get('risk_level', 'UNKNOWN')}
        """
        
        self.console.print(Panel(summary_text, title="Analysis Complete", border_style="green"))
    
    def _print_complexity_distribution(self, distribution: Dict) -> None:
        """Print complexity distribution chart."""
        self.console.print("\n[bold]Complexity Distribution:[/bold]")
        
        for level, percentage in distribution.items():
            bar_length = int(percentage / 5)  # Scale for display
            bar = "█" * bar_length + "░" * (20 - bar_length)
            self.console.print(f"  {level.capitalize()}: {bar} {percentage:.1f}%")
    
    def _get_complexity_status(self, complexity: float) -> str:
        """Get status emoji for complexity."""
        if complexity <= 5:
            return "✅"
        elif complexity <= 10:
            return "⚠️"
        elif complexity <= 15:
            return "🔶"
        else:
            return "🔴"
    
    def _get_maintainability_status(self, maintainability: float) -> str:
        """Get status emoji for maintainability."""
        if maintainability >= 80:
            return "✅"
        elif maintainability >= 60:
            return "⚠️"
        elif maintainability >= 40:
            return "🔶"
        else:
            return "🔴"
    
    def _get_risk_status(self, risk_level: str) -> str:
        """Get status emoji for risk level."""
        risk_map = {
            'LOW': '✅',
            'MEDIUM': '⚠️',
            'HIGH': '🔶',
            'CRITICAL': '🔴'
        }
        return risk_map.get(risk_level, '❓')
    
    def _generate_html_content(self, result: AnalysisResult) -> str:
        """Generate complete HTML report content."""
        # Create interactive visualizations
        complexity_chart = self._create_complexity_chart(result)
        maintainability_chart = self._create_maintainability_chart(result)
        risk_hotspots_chart = self._create_risk_hotspots_chart(result)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CodeSage Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    color: #7f8c8d;
                    margin: 10px 0 0 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .metric-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 1.2em;
                }}
                .metric-card .value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .insights {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .insights h3 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                .insights ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .insights li {{
                    margin-bottom: 8px;
                    color: #34495e;
                }}
                .hotspots {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .hotspots h3 {{
                    color: #856404;
                    margin-top: 0;
                }}
                .hotspot-item {{
                    background: white;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #dc3545;
                }}
                .hotspot-item strong {{
                    color: #dc3545;
                }}
                .chart-container {{
                    margin: 20px 0;
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 20px;
                    background: white;
                }}
                .suggestions {{
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 10px;
                    padding: 20px;
                }}
                .suggestions h3 {{
                    color: #155724;
                    margin-top: 0;
                }}
                .suggestions ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .suggestions li {{
                    margin-bottom: 8px;
                    color: #155724;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #e0e0e0;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧩 CodeSage Analysis Report</h1>
                    <p>AI-Enhanced Code Complexity Analysis</p>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Total Files</h3>
                        <div class="value">{result.project_metrics.get('total_files', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Functions</h3>
                        <div class="value">{result.project_metrics.get('total_functions', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Avg Complexity</h3>
                        <div class="value">{result.project_metrics.get('average_complexity', 0):.1f}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Maintainability</h3>
                        <div class="value">{result.project_metrics.get('average_maintainability', 100):.1f}/100</div>
                    </div>
                    <div class="metric-card">
                        <h3>Risk Level</h3>
                        <div class="value">{result.project_metrics.get('risk_level', 'UNKNOWN')}</div>
                    </div>
                    <div class="metric-card">
                        <h3>AI Anomaly Score</h3>
                        <div class="value">{result.project_metrics.get('average_ai_anomaly_score', 0):.1f}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🤖 AI Insights</h2>
                    <div class="insights">
                        <h3>Intelligent Analysis Results</h3>
                        <ul>
                            {''.join(f'<li>{insight}</li>' for insight in result.ai_insights)}
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Complexity Analysis</h2>
                    <div class="chart-container">
                        {complexity_chart}
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Maintainability Overview</h2>
                    <div class="chart-container">
                        {maintainability_chart}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔥 Risk Hotspots</h2>
                    <div class="hotspots">
                        <h3>High-Risk Areas Identified by AI</h3>
                        {''.join(f'''
                        <div class="hotspot-item">
                            <strong>{hotspot['file']} - {hotspot['function']}</strong><br>
                            Risk Score: {hotspot['risk_score']:.1f} | 
                            Complexity: {hotspot['complexity']} | 
                            Lines: {hotspot['lines']} | 
                            Type: {hotspot['risk_type']}
                        </div>
                        ''' for hotspot in result.risk_hotspots[:5])}
                    </div>
                    <div class="chart-container">
                        {risk_hotspots_chart}
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 Improvement Suggestions</h2>
                    <div class="suggestions">
                        <h3>AI-Generated Recommendations</h3>
                        <ul>
                            {''.join(f'<li>{suggestion}</li>' for suggestion in result.suggestions)}
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by CodeSage - AI-Enhanced Code Complexity Estimator</p>
                    <p>For more information, visit the CodeSage documentation</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_complexity_chart(self, result: AnalysisResult) -> str:
        """Create complexity distribution chart."""
        if not result.files:
            return "<p>No data available for complexity chart.</p>"
        
        # Prepare data
        complexities = []
        for file_metric in result.files:
            for func_metric in file_metric.functions:
                complexities.append(func_metric.cyclomatic_complexity)
        
        if not complexities:
            return "<p>No function complexity data available.</p>"
        
        # Create histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=complexities,
            nbinsx=20,
            name='Function Complexity',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Distribution of Cyclomatic Complexity',
            xaxis_title='Cyclomatic Complexity',
            yaxis_title='Number of Functions',
            showlegend=False,
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_maintainability_chart(self, result: AnalysisResult) -> str:
        """Create maintainability chart."""
        if not result.files:
            return "<p>No data available for maintainability chart.</p>"
        
        # Prepare data
        files = [f.filename for f in result.files]
        maintainability_scores = [f.maintainability_index for f in result.files]
        ai_scores = [f.ai_anomaly_score for f in result.files]
        
        fig = go.Figure()
        
        # Maintainability scores
        fig.add_trace(go.Bar(
            x=files,
            y=maintainability_scores,
            name='Maintainability Index',
            marker_color='#2ecc71',
            opacity=0.7
        ))
        
        # AI anomaly scores (inverted for better visualization)
        fig.add_trace(go.Bar(
            x=files,
            y=[100 - score for score in ai_scores],
            name='AI Quality Score (Inverted)',
            marker_color='#e74c3c',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='File Maintainability vs AI Quality Score',
            xaxis_title='Files',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_risk_hotspots_chart(self, result: AnalysisResult) -> str:
        """Create risk hotspots visualization."""
        if not result.risk_hotspots:
            return "<p>No risk hotspots identified.</p>"
        
        # Prepare data
        hotspots = result.risk_hotspots[:10]  # Top 10
        labels = [f"{h['file']}\n{h['function']}" for h in hotspots]
        risk_scores = [h['risk_score'] for h in hotspots]
        complexities = [h['complexity'] for h in hotspots]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=complexities,
            y=risk_scores,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(
                size=risk_scores,
                color=risk_scores,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            name='Risk Hotspots'
        ))
        
        fig.update_layout(
            title='Risk Hotspots Analysis',
            xaxis_title='Cyclomatic Complexity',
            yaxis_title='AI Risk Score',
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

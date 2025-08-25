"""
Command Line Interface Module

Provides the main CLI interface for CodeSage with argument parsing
and user-friendly output formatting.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .analyzer import CodeAnalyzer
from .reporter import ReportGenerator


def main():
    """Main entry point for CodeSage CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Display welcome message
        if not args.quiet:
            display_welcome_message(console)
        
        # Validate input path
        input_path = Path(args.path)
        if not input_path.exists():
            console.print(f"[red]Error: Path '{args.path}' does not exist.[/red]")
            sys.exit(1)
        
        # Initialize analyzer and reporter
        analyzer = CodeAnalyzer()
        reporter = ReportGenerator()
        
        # Perform analysis with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=args.quiet
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)
            
            try:
                result = analyzer.analyze_project(
                    str(input_path),
                    train_ml=not args.no_ml
                )
                progress.update(task, description="Analysis complete!")
            except Exception as e:
                console.print(f"[red]Error during analysis: {e}[/red]")
                sys.exit(1)
        
        # Generate reports
        if args.html:
            html_path = reporter.generate_html_report(result, args.html_output)
            if not args.quiet:
                console.print(f"\n[green]HTML report generated: {html_path}[/green]")
        
        # Display CLI report
        if not args.quiet:
            reporter.generate_cli_report(result, detailed=args.detailed)
        
        # Exit with appropriate code based on risk level
        risk_level = result.project_metrics.get('risk_level', 'UNKNOWN')
        if args.strict and risk_level in ['HIGH', 'CRITICAL']:
            console.print(f"\n[yellow]Warning: High risk level detected ({risk_level}). Exiting with code 1.[/yellow]")
            sys.exit(1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="codesage",
        description="🧩 CodeSage - AI-Enhanced Code Complexity Estimator",
        epilog="For more information, visit the CodeSage documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument(
        "path",
        help="Path to Python file or directory to analyze"
    )
    
    # Output options
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report in addition to CLI output"
    )
    
    parser.add_argument(
        "--html-output",
        default="codesage_report.html",
        help="Output path for HTML report (default: codesage_report.html)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis for each file"
    )
    
    # Analysis options
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable machine learning enhancements (use rule-based analysis only)"
    )
    
    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed error information"
    )
    
    # Quality gates
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if high risk levels are detected"
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="CodeSage 0.1.0"
    )
    
    return parser


def display_welcome_message(console: Console) -> None:
    """Display the CodeSage welcome message."""
    welcome_text = Text()
    welcome_text.append("🧩 ", style="bold blue")
    welcome_text.append("CodeSage", style="bold blue")
    welcome_text.append(" - AI-Enhanced Code Complexity Estimator", style="blue")
    
    subtitle = Text("Analyzing your code with machine learning intelligence...", style="dim")
    
    panel = Panel(
        subtitle,
        title=welcome_text,
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)
    console.print()


def run_analysis(
    path: str,
    html_output: Optional[str] = None,
    detailed: bool = False,
    train_ml: bool = True,
    quiet: bool = False
) -> dict:
    """
    Run CodeSage analysis programmatically.
    
    Args:
        path: Path to Python file or directory
        html_output: Optional path for HTML report
        detailed: Whether to include detailed analysis
        train_ml: Whether to use ML enhancements
        quiet: Whether to suppress output
    
    Returns:
        Dictionary containing analysis results
    """
    console = Console()
    
    try:
        # Validate input path
        input_path = Path(path)
        if not input_path.exists():
            raise ValueError(f"Path '{path}' does not exist")
        
        # Initialize analyzer and reporter
        analyzer = CodeAnalyzer()
        reporter = ReportGenerator()
        
        # Perform analysis
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing code...", total=None)
                result = analyzer.analyze_project(str(input_path), train_ml=train_ml)
                progress.update(task, description="Analysis complete!")
        else:
            result = analyzer.analyze_project(str(input_path), train_ml=train_ml)
        
        # Generate HTML report if requested
        if html_output:
            html_path = reporter.generate_html_report(result, html_output)
            if not quiet:
                console.print(f"HTML report generated: {html_path}")
        
        # Display CLI report if not quiet
        if not quiet:
            reporter.generate_cli_report(result, detailed=detailed)
        
        # Return results as dictionary
        return {
            'files': [
                {
                    'filename': f.filename,
                    'total_lines': f.total_lines,
                    'total_functions': f.total_functions,
                    'average_complexity': f.average_complexity,
                    'maintainability_index': f.maintainability_index,
                    'ai_anomaly_score': f.ai_anomaly_score,
                    'functions': [
                        {
                            'name': func.name,
                            'cyclomatic_complexity': func.cyclomatic_complexity,
                            'lines_of_code': func.lines_of_code,
                            'nesting_depth': func.nesting_depth,
                            'parameters': func.parameters,
                            'maintainability_index': func.maintainability_index,
                            'ai_risk_score': func.ai_risk_score
                        }
                        for func in f.functions
                    ]
                }
                for f in result.files
            ],
            'project_metrics': result.project_metrics,
            'suggestions': result.suggestions,
            'ai_insights': result.ai_insights,
            'risk_hotspots': result.risk_hotspots
        }
        
    except Exception as e:
        if not quiet:
            console.print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

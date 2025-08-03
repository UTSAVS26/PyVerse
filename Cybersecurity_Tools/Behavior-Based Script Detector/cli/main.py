"""
Main CLI interface for Behavior-Based Script Detector

This module provides the command-line interface for analyzing
Python scripts for suspicious behavior patterns.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer.pattern_rules import PatternRules
from analyzer.score_calculator import ScoreCalculator
from analyzer.report_generator import ReportGenerator
from utils.file_loader import FileLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel


class BehaviorAnalyzer:
    """Main analyzer class that coordinates the analysis process."""
    
    def __init__(self):
        """Initialize the analyzer with all components."""
        self.pattern_rules = PatternRules()
        self.score_calculator = ScoreCalculator()
        self.report_generator = ReportGenerator()
        self.file_loader = FileLoader()
        self.console = Console()
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Load and parse file
            file_content, tree, atok = self.file_loader.load_file(file_path)
            
            # Analyze for patterns
            findings = self.pattern_rules.analyze_ast(tree, atok)
            
            # Calculate risk score
            score_result = self.score_calculator.calculate_risk_score(findings)
            
            # Get risk assessment
            assessment = self.score_calculator.get_risk_assessment(
                score_result['risk_score'], findings
            )
            
            # Combine results
            result = {
                'filename': file_path,
                'findings': findings,
                'risk_score': score_result['risk_score'],
                'verdict': score_result['verdict'],
                'total_findings': score_result['total_findings'],
                'severity_breakdown': score_result['severity_breakdown'],
                'assessment': assessment
            }
            
            return result
            
        except Exception as e:
            return {
                'filename': file_path,
                'error': str(e),
                'risk_score': 0,
                'verdict': 'Error',
                'total_findings': 0,
                'severity_breakdown': {},
                'assessment': {'summary': f'Error analyzing file: {e}'}
            }
    
    def analyze_directory(self, directory_path: str, threshold: int = 0) -> List[Dict[str, Any]]:
        """
        Analyze all Python files in a directory.
        
        Args:
            directory_path: Path to the directory
            threshold: Minimum risk score to include in results
            
        Returns:
            List of analysis results
        """
        results = []
        
        try:
            # Get directory stats
            stats = self.file_loader.get_directory_stats(directory_path)
            if 'error' in stats:
                self.console.print(f"[red]Error: {stats['error']}[/red]")
                return results
            
            self.console.print(f"[blue]Found {stats['python_files']} Python files to analyze[/blue]")
            
            # Load all files
            files_data = self.file_loader.load_directory(directory_path)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Analyzing files...", total=len(files_data))
                
                for file_path, file_content, tree, atok in files_data:
                    progress.update(task, description=f"Analyzing {os.path.basename(file_path)}")
                    
                    # Analyze file
                    findings = self.pattern_rules.analyze_ast(tree, atok)
                    score_result = self.score_calculator.calculate_risk_score(findings)
                    
                    # Check threshold
                    if score_result['risk_score'] >= threshold:
                        assessment = self.score_calculator.get_risk_assessment(
                            score_result['risk_score'], findings
                        )
                        
                        result = {
                            'filename': file_path,
                            'findings': findings,
                            'risk_score': score_result['risk_score'],
                            'verdict': score_result['verdict'],
                            'total_findings': score_result['total_findings'],
                            'severity_breakdown': score_result['severity_breakdown'],
                            'assessment': assessment
                        }
                        
                        results.append(result)
                    
                    progress.advance(task)
            
        except Exception as e:
            self.console.print(f"[red]Error analyzing directory: {e}[/red]")
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Behavior-Based Script Detector - Analyze Python scripts for suspicious behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file suspicious_script.py
  python main.py --file script.py --report reports/
  python main.py --dir ./downloads/ --threshold 60
  python main.py --file script.py --json --markdown
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        help='Path to Python file to analyze'
    )
    input_group.add_argument(
        '--dir', '-d',
        help='Path to directory containing Python files to analyze'
    )
    
    # Output options
    parser.add_argument(
        '--report', '-r',
        help='Directory to save reports (default: reports/)',
        default='reports/'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Generate JSON report'
    )
    parser.add_argument(
        '--markdown', '--md',
        action='store_true',
        help='Generate Markdown report'
    )
    parser.add_argument(
        '--no-console',
        action='store_true',
        help='Suppress console output'
    )
    
    # Analysis options
    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=0,
        help='Minimum risk score to include in directory analysis (default: 0)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BehaviorAnalyzer()
    console = Console()
    
    try:
        if args.file:
            # Analyze single file
            console.print(f"[blue]Analyzing file: {args.file}[/blue]")
            
            result = analyzer.analyze_file(args.file)
            
            if 'error' in result:
                console.print(f"[red]Error: {result['error']}[/red]")
                sys.exit(1)
            
            # Display results
            if not args.no_console:
                analyzer.report_generator.generate_console_report(args.file, result)
            
            # Generate reports
            if args.json or args.markdown or args.report:
                if args.json:
                    json_report = analyzer.report_generator.generate_json_report(args.file, result)
                    report_path = analyzer.report_generator.save_report(
                        json_report, args.file, "json", args.report
                    )
                    console.print(f"[green]JSON report saved: {report_path}[/green]")
                
                if args.markdown:
                    md_report = analyzer.report_generator.generate_markdown_report(args.file, result)
                    report_path = analyzer.report_generator.save_report(
                        md_report, args.file, "md", args.report
                    )
                    console.print(f"[green]Markdown report saved: {report_path}[/green]")
        
        elif args.dir:
            # Analyze directory
            console.print(f"[blue]Analyzing directory: {args.dir}[/blue]")
            
            results = analyzer.analyze_directory(args.dir, args.threshold)
            
            if not results:
                console.print("[yellow]No files found or all files below threshold[/yellow]")
                return
            
            # Display summary
            console.print(f"\n[bold]Analysis Summary:[/bold]")
            console.print(f"Files analyzed: {len(results)}")
            
            high_risk = sum(1 for r in results if 'High Risk' in r['verdict'] or 'Critical Risk' in r['verdict'])
            medium_risk = sum(1 for r in results if 'Medium Risk' in r['verdict'])
            low_risk = sum(1 for r in results if 'Low Risk' in r['verdict'])
            
            console.print(f"High/Critical Risk: {high_risk}")
            console.print(f"Medium Risk: {medium_risk}")
            console.print(f"Low Risk: {low_risk}")
            
            # Generate batch report
            if args.json or args.markdown or args.report:
                batch_report = analyzer.report_generator.generate_batch_report(results)
                report_path = analyzer.report_generator.save_report(
                    batch_report, f"batch_analysis_{len(results)}_files", "json", args.report
                )
                console.print(f"[green]Batch report saved: {report_path}[/green]")
                
                # Generate individual reports
                for result in results:
                    if args.json:
                        json_report = analyzer.report_generator.generate_json_report(
                            result['filename'], result
                        )
                        analyzer.report_generator.save_report(
                            json_report, result['filename'], "json", args.report
                        )
                    
                    if args.markdown:
                        md_report = analyzer.report_generator.generate_markdown_report(
                            result['filename'], result
                        )
                        analyzer.report_generator.save_report(
                            md_report, result['filename'], "md", args.report
                        )
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 
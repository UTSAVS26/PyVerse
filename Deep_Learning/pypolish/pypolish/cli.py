"""
Command Line Interface for PyPolish

Provides a CLI for cleaning Python code files.
"""

import click
import sys
from pathlib import Path
from typing import Optional
from .code_cleaner import CodeCleaner
from .diff_viewer import DiffViewer


@click.group()
@click.version_option(version="0.1.0", prog_name="PyPolish")
def main():
    """🧹 PyPolish - AI Code Cleaner and Rewriter
    
    Transform raw Python scripts into clean, optimized, and more Pythonic versions.
    """
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
@click.option('--no-analysis', is_flag=True, help='Skip analysis display')
@click.option('--no-diff', is_flag=True, help='Skip diff display')
@click.option('--quiet', '-q', is_flag=True, help='Suppress all output except errors')
def clean(input_file: Path, output: Optional[Path], line_length: int, 
          no_analysis: bool, no_diff: bool, quiet: bool):
    """Clean a Python file and optionally save the result."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        
        if not quiet:
            click.echo(f"🧹 Cleaning {input_file}...")
        
        cleaned_code, analysis_results = cleaner.clean_file(
            str(input_file),
            str(output) if output else None,
            show_analysis=not no_analysis and not quiet,
            show_diff=not no_diff and not quiet
        )
        
        if 'error' in analysis_results:
            click.echo(f"❌ Error: {analysis_results['error']}", err=True)
            sys.exit(1)
        
        if not quiet:
            if output:
                click.echo(f"✅ Cleaned code saved to: {output}")
            else:
                click.echo("✅ Code cleaning completed!")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
def diff(input_file: Path, line_length: int):
    """Show diff between original and cleaned code without saving."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        diff_viewer = DiffViewer()
        
        click.echo(f"🔍 Analyzing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        cleaned_code, analysis_results = cleaner.clean_code(
            original_code, 
            show_analysis=False, 
            show_diff=False
        )
        
        if 'error' in analysis_results:
            click.echo(f"❌ Error: {analysis_results['error']}", err=True)
            sys.exit(1)
        
        if original_code == cleaned_code:
            click.echo("✅ No changes needed - code is already clean!")
        else:
            diff_viewer.show_diff(original_code, cleaned_code)
            diff_viewer.show_statistics(original_code, cleaned_code)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
def analyze(input_file: Path, line_length: int):
    """Analyze a Python file and show detailed analysis without cleaning."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        diff_viewer = DiffViewer()
        
        click.echo(f"🔍 Analyzing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        analysis_results = cleaner.analyzer.analyze(code)
        
        if 'error' in analysis_results:
            click.echo(f"❌ Error: {analysis_results['error']}", err=True)
            sys.exit(1)
        
        # Show analysis summary
        diff_viewer.show_analysis_summary(analysis_results)
        
        # Show code metrics
        metrics = cleaner.get_code_metrics(code)
        if 'error' not in metrics:
            click.echo(f"\n📊 Code Metrics:")
            click.echo(f"   Functions: {metrics['function_count']}")
            click.echo(f"   Classes: {metrics['class_count']}")
            click.echo(f"   Total Lines: {metrics['total_lines']}")
            click.echo(f"   Total Characters: {metrics['total_chars']}")
            click.echo(f"   Unused Imports: {metrics['unused_imports']}")
            click.echo(f"   Undefined Names: {metrics['undefined_names']}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
def side_by_side(input_file: Path, line_length: int):
    """Show original and cleaned code side by side."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        diff_viewer = DiffViewer()
        
        click.echo(f"🔍 Processing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        cleaned_code, analysis_results = cleaner.clean_code(
            original_code, 
            show_analysis=False, 
            show_diff=False
        )
        
        if 'error' in analysis_results:
            click.echo(f"❌ Error: {analysis_results['error']}", err=True)
            sys.exit(1)
        
        diff_viewer.show_side_by_side(original_code, cleaned_code)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
def validate(input_file: Path, line_length: int):
    """Validate that a Python file is syntactically correct."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        if cleaner.validate_code(code):
            click.echo(f"✅ {input_file} is valid Python code!")
        else:
            click.echo(f"❌ {input_file} contains syntax errors!", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--line-length', '-l', default=88, help='Maximum line length (default: 88)')
def format_only(input_file: Path, output: Optional[Path], line_length: int):
    """Format code only (no analysis or improvements)."""
    try:
        cleaner = CodeCleaner(line_length=line_length)
        
        click.echo(f"🎨 Formatting {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        # Only format, no analysis or improvements
        cleaned_code = cleaner.formatter.format_code(original_code)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            click.echo(f"✅ Formatted code saved to: {output}")
        else:
            click.echo(cleaned_code)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

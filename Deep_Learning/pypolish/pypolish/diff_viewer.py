"""
Diff Viewer for PyPolish

Shows before/after differences in code with rich formatting.
"""

import difflib
from typing import List, Tuple
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class DiffViewer:
    """Displays code differences in a rich, formatted way."""
    
    def __init__(self):
        self.console = Console()
    
    def show_diff(self, original_code: str, cleaned_code: str, title: str = "Code Changes") -> None:
        """Show a rich diff between original and cleaned code."""
        if original_code == cleaned_code:
            self.console.print(Panel("✅ No changes needed - code is already clean!", style="green"))
            return
        
        # Create diff
        diff_lines = self._generate_diff(original_code, cleaned_code)
        
        # Display diff
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        self.console.print("=" * 50)
        
        for line in diff_lines:
            if line.startswith('+'):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith('-'):
                self.console.print(f"[red]{line}[/red]")
            elif line.startswith('@'):
                self.console.print(f"[yellow]{line}[/yellow]")
            else:
                self.console.print(line)
    
    def show_side_by_side(self, original_code: str, cleaned_code: str, title: str = "Before vs After") -> None:
        """Show original and cleaned code side by side."""
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        
        # Create a table for side-by-side comparison
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Before", style="red", width=50)
        table.add_column("After", style="green", width=50)
        
        # Split code into lines
        original_lines = original_code.split('\n')
        cleaned_lines = cleaned_code.split('\n')
        
        # Pad shorter list with empty strings
        max_lines = max(len(original_lines), len(cleaned_lines))
        original_lines.extend([''] * (max_lines - len(original_lines)))
        cleaned_lines.extend([''] * (max_lines - len(cleaned_lines)))
        
        # Add lines to table
        for i, (orig_line, clean_line) in enumerate(zip(original_lines, cleaned_lines)):
            table.add_row(
                f"{i+1:3d}: {orig_line}",
                f"{i+1:3d}: {clean_line}"
            )
        
        self.console.print(table)
    
    def show_analysis_summary(self, analysis_results: dict) -> None:
        """Show a summary of the analysis results."""
        issues = analysis_results.get('issues', [])
        suggestions = analysis_results.get('suggestions', [])
        
        # Create summary table
        table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="yellow")
        table.add_column("Severity", style="red")
        
        # Count issues by severity
        severity_counts = {}
        for issue in issues:
            severity = issue.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Add rows to table
        for severity, count in severity_counts.items():
            table.add_row("Issue", str(count), severity.title())
        
        table.add_row("Suggestion", str(len(suggestions)), "Info")
        
        self.console.print(table)
        
        # Show detailed issues and suggestions
        if issues:
            self.console.print("\n[bold red]Issues Found:[/bold red]")
            for issue in issues:
                self.console.print(f"• [red]{issue['message']}[/red] (Line {issue.get('line', 'N/A')})")
        
        if suggestions:
            self.console.print("\n[bold yellow]Suggestions:[/bold yellow]")
            for suggestion in suggestions:
                self.console.print(f"• [yellow]{suggestion['message']}[/yellow] (Line {suggestion.get('line', 'N/A')})")
    
    def show_syntax_highlighted(self, code: str, language: str = "python", title: str = "Code") -> None:
        """Show syntax highlighted code."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        panel = Panel(syntax, title=title, border_style="blue")
        self.console.print(panel)
    
    def show_before_after_highlighted(self, original_code: str, cleaned_code: str) -> None:
        """Show before and after code with syntax highlighting."""
        self.console.print("\n[bold red]Before:[/bold red]")
        self.show_syntax_highlighted(original_code, title="Original Code")
        
        self.console.print("\n[bold green]After:[/bold green]")
        self.show_syntax_highlighted(cleaned_code, title="Cleaned Code")
    
    def _generate_diff(self, original_code: str, cleaned_code: str) -> List[str]:
        """Generate unified diff between original and cleaned code."""
        original_lines = original_code.splitlines(keepends=True)
        cleaned_lines = cleaned_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            cleaned_lines,
            fromfile='original.py',
            tofile='cleaned.py',
            lineterm=''
        )
        
        return list(diff)
    
    def show_statistics(self, original_code: str, cleaned_code: str) -> None:
        """Show statistics about the code changes."""
        original_lines = len(original_code.split('\n'))
        cleaned_lines = len(cleaned_code.split('\n'))
        original_chars = len(original_code)
        cleaned_chars = len(cleaned_code)
        
        # Calculate differences
        line_diff = cleaned_lines - original_lines
        char_diff = cleaned_chars - original_chars
        
        # Create statistics table
        table = Table(title="Code Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Original", style="red")
        table.add_column("Cleaned", style="green")
        table.add_column("Difference", style="yellow")
        
        table.add_row("Lines", str(original_lines), str(cleaned_lines), f"{line_diff:+d}")
        table.add_row("Characters", str(original_chars), str(cleaned_chars), f"{char_diff:+d}")
        
        # Calculate improvement percentage
        if original_chars > 0:
            improvement = ((cleaned_chars - original_chars) / original_chars) * 100
            table.add_row("Size Change", "", "", f"{improvement:+.1f}%")
        
        self.console.print(table)
    
    def show_improvements_list(self, analysis_results: dict) -> None:
        """Show a list of improvements made to the code."""
        improvements = []
        
        # Extract improvements from analysis
        for suggestion in analysis_results.get('suggestions', []):
            if suggestion.get('type') in ['missing_type_hint', 'missing_docstring', 'ternary_expression']:
                improvements.append(suggestion['message'])
        
        for issue in analysis_results.get('issues', []):
            if issue.get('type') in ['long_function', 'infinite_loop']:
                improvements.append(f"Fixed: {issue['message']}")
        
        if improvements:
            self.console.print("\n[bold green]Improvements Made:[/bold green]")
            for i, improvement in enumerate(improvements, 1):
                self.console.print(f"{i}. [green]{improvement}[/green]")
        else:
            self.console.print("\n[bold yellow]No specific improvements identified.[/bold yellow]")

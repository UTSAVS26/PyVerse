"""
Command-line interface for ExplainDoku
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text

from ..core.grid import Grid
from ..core.solver import Solver
from ..explain.trace import TraceRecorder
from ..explain.verbalizer import Verbalizer


class ExplainDokuCLI:
    """Command-line interface for ExplainDoku"""
    
    def __init__(self):
        self.console = Console()
        self.verbalizer = Verbalizer()
    
    def main(self):
        """Main CLI entry point"""
        parser = self._create_parser()
        args = parser.parse_args()
        
        try:
            if args.command == "solve":
                self.solve_command(args)
            elif args.command == "step":
                self.step_command(args)
            elif args.command == "help":
                self.help_command(args)
            else:
                parser.print_help()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            prog="exdoku",
            description="🧩 ExplainDoku - Sudoku Solver with Human-Style Explanations",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  exdoku solve --grid "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
  exdoku step --file examples/medium.txt
  exdoku solve --grid "..." --no-search --explain
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Solve command
        solve_parser = subparsers.add_parser("solve", help="Solve a Sudoku puzzle")
        solve_parser.add_argument("--grid", "-g", help="81-character grid string")
        solve_parser.add_argument("--file", "-f", help="File containing grid string")
        solve_parser.add_argument("--no-search", action="store_true", help="Don't use search/backtracking")
        solve_parser.add_argument("--explain", "-e", action="store_true", help="Show detailed explanations")
        solve_parser.add_argument("--max-backtracks", type=int, default=10000, help="Maximum backtracks for search")
        
        # Step command
        step_parser = subparsers.add_parser("step", help="Solve step by step")
        step_parser.add_argument("--grid", "-g", help="81-character grid string")
        step_parser.add_argument("--file", "-f", help="File containing grid string")
        step_parser.add_argument("--auto", action="store_true", help="Auto-advance through steps")
        
        # Help command
        help_parser = subparsers.add_parser("help", help="Show help for techniques")
        help_parser.add_argument("technique", nargs="?", help="Specific technique to explain")
        
        return parser
    
    def solve_command(self, args):
        """Handle solve command"""
        # Get grid
        grid = self._get_grid_from_args(args)
        if grid is None:
            self.console.print("[red]Error: No grid provided. Use --grid or --file.[/red]")
            return
        
        # Display initial grid
        self.console.print(Panel(
            self._format_grid(grid),
            title="🧩 Initial Puzzle",
            border_style="blue"
        ))
        
        # Solve with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Solving puzzle...", total=None)
            
            # Create solver and recorder
            solver = Solver(grid)
            recorder = TraceRecorder()
            
            # Solve
            result = solver.solve(use_search=not args.no_search, max_backtracks=args.max_backtracks)
            
            progress.update(task, description="Generating explanations...")
        
        # Display results
        self._display_solve_result(result, args.explain)
    
    def step_command(self, args):
        """Handle step command"""
        # Get grid
        grid = self._get_grid_from_args(args)
        if grid is None:
            self.console.print("[red]Error: No grid provided. Use --grid or --file.[/red]")
            return
        
        # Display initial grid
        self.console.print(Panel(
            self._format_grid(grid),
            title="🧩 Initial Puzzle",
            border_style="blue"
        ))
        
        # Create solver
        solver = Solver(grid)
        current_grid = grid.copy()
        
        step_number = 1
        while not current_grid.is_solved():
            # Get next step
            next_step = solver._get_next_human_step()
            if next_step is None:
                # Try search step
                next_step = solver._get_next_search_step()
                if next_step is None:
                    self.console.print("[yellow]No more steps possible.[/yellow]")
                    break
            
            # Display step
            self.console.print(f"\n[bold cyan]Step {step_number}[/bold cyan]")
            self.console.print(f"Technique: [bold]{next_step.technique.replace('_', ' ').title()}[/bold]")
            self.console.print(f"Explanation: {next_step.explanation}")
            
            if next_step.cell_position and next_step.value:
                row, col = next_step.cell_position
                self.console.print(f"Action: Place {next_step.value} at R{row+1}C{col+1}")
            
            # Show eliminations if any
            if next_step.eliminations:
                elim_text = "Eliminations: "
                elims = []
                for (row, col), digit in next_step.eliminations:
                    elims.append(f"{digit} from R{row+1}C{col+1}")
                elim_text += ", ".join(elims)
                self.console.print(elim_text)
            
            # Update grid
            if next_step.cell_position and next_step.value:
                row, col = next_step.cell_position
                current_grid.set_value(row, col, next_step.value)
            
            # Display updated grid
            self.console.print(Panel(
                self._format_grid(current_grid),
                title=f"After Step {step_number}",
                border_style="green"
            ))
            
            step_number += 1
            
            # Ask to continue (unless auto mode)
            if not args.auto:
                if not Confirm.ask("Continue to next step?"):
                    break
        
        # Final result
        if current_grid.is_solved():
            self.console.print("[bold green]✅ Puzzle solved![/bold green]")
        else:
            self.console.print("[yellow]Puzzle not fully solved.[/yellow]")
    
    def help_command(self, args):
        """Handle help command"""
        if args.technique:
            self._show_technique_help(args.technique)
        else:
            self._show_general_help()
    
    def _get_grid_from_args(self, args) -> Grid:
        """Get grid from command line arguments"""
        grid_str = None
        
        if args.grid:
            grid_str = args.grid
        elif args.file:
            try:
                with open(args.file, 'r') as f:
                    grid_str = f.read().strip()
            except FileNotFoundError:
                self.console.print(f"[red]Error: File '{args.file}' not found.[/red]")
                return None
            except Exception as e:
                self.console.print(f"[red]Error reading file: {e}[/red]")
                return None
        
        if grid_str is None:
            return None
        
        try:
            return Grid.from_string(grid_str)
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return None
    
    def _format_grid(self, grid: Grid) -> str:
        """Format grid for display"""
        return grid.to_display_string()
    
    def _display_solve_result(self, result, show_explanation: bool = False):
        """Display solve result"""
        # Create summary table
        table = Table(title="🧩 Solve Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Success", "✅ Yes" if result.success else "❌ No")
        table.add_row("Total Steps", str(result.total_steps))
        table.add_row("Human Steps", str(result.human_steps))
        table.add_row("Search Steps", str(result.search_steps))
        table.add_row("Backtracks", str(result.backtrack_count))
        
        self.console.print(table)
        
        # Show final grid if solved
        if result.success and result.final_grid:
            self.console.print(Panel(
                self._format_grid(result.final_grid),
                title="✅ Solved Puzzle",
                border_style="green"
            ))
        
        # Show detailed explanation if requested
        if show_explanation and result.steps:
            self.console.print("\n[bold]Detailed Steps:[/bold]")
            for i, step in enumerate(result.steps, 1):
                self.console.print(f"{i}. {step.explanation}")
    
    def _show_technique_help(self, technique: str):
        """Show help for a specific technique"""
        help_info = self.verbalizer.get_technique_help(technique)
        
        self.console.print(Panel(
            f"[bold]{help_info['name']}[/bold]\n"
            f"Category: {help_info['category']}\n"
            f"Difficulty: {help_info['difficulty']}\n\n"
            f"{help_info['description']}\n\n"
            f"Template: {help_info['template']}",
            title=f"Help: {help_info['name']}",
            border_style="blue"
        ))
    
    def _show_general_help(self):
        """Show general help"""
        techniques = self.verbalizer.templates.get_all_techniques()
        
        # Group by category
        categories = {}
        for technique in techniques:
            category = self.verbalizer._get_technique_category(technique)
            if category not in categories:
                categories[category] = []
            categories[category].append(technique)
        
        # Display help
        self.console.print("[bold]Available Techniques:[/bold]\n")
        
        for category, techs in categories.items():
            self.console.print(f"[bold cyan]{category}:[/bold cyan]")
            for technique in techs:
                difficulty = self.verbalizer.templates.get_technique_difficulty(technique)
                description = self.verbalizer.templates.get_technique_description(technique)
                self.console.print(f"  • {technique.replace('_', ' ').title()} ({difficulty}): {description}")
            self.console.print()
        
        self.console.print("Use 'exdoku help <technique>' for detailed help on a specific technique.")


def main():
    """Main entry point"""
    cli = ExplainDokuCLI()
    cli.main()


if __name__ == "__main__":
    main()

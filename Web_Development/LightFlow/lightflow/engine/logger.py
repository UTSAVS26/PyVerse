"""
Logger - Rich logging interface for LightFlow framework.
"""

import logging
import os
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class Logger:
    """Rich logging interface for LightFlow."""
    
    def __init__(self, log_dir: str = "logs", level: str = "INFO"):
        self.log_dir = log_dir
        self.console = Console()
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging(level)
        
    def _setup_logging(self, level: str):
        """Setup logging configuration."""
        # Create logger
        self.logger = logging.getLogger("lightflow")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True
        )
        rich_handler.setLevel(getattr(logging, level.upper()))
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"lightflow_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(rich_handler)
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
        
    def success(self, message: str):
        """Log success message with green color."""
        self.console.print(f"[green]✓ {message}[/green]")
        
    def task_start(self, task_name: str):
        """Log task start."""
        self.console.print(f"[blue]▶ Starting task: {task_name}[/blue]")
        
    def task_success(self, task_name: str, duration: float):
        """Log task success."""
        self.console.print(f"[green]✓ Task completed: {task_name} ({duration:.2f}s)[/green]")
        
    def task_failed(self, task_name: str, error: str, duration: float):
        """Log task failure."""
        self.console.print(f"[red]✗ Task failed: {task_name} ({duration:.2f}s)[/red]")
        self.console.print(f"[red]Error: {error}[/red]")
        
    def workflow_start(self, workflow_name: str, task_count: int):
        """Log workflow start."""
        self.console.print(Panel(
            f"[bold blue]LightFlow Workflow[/bold blue]\n"
            f"Workflow: {workflow_name}\n"
            f"Tasks: {task_count}",
            border_style="blue"
        ))
        
    def workflow_complete(self, workflow_name: str, results: dict):
        """Log workflow completion."""
        completed = sum(1 for r in results.values() if getattr(r, 'success', False))
        failed = len(results) - completed
        
        self.console.print(Panel(
            f"[bold green]Workflow Complete[/bold green]\n"
            f"Workflow: {workflow_name}\n"
            f"Completed: {completed}\n"
            f"Failed: {failed}",
            border_style="green" if failed == 0 else "red"
        ))
        
    def print_task_results(self, results: dict):
        """Print task results in a table."""
        table = Table(title="Task Results")
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Exit Code", style="yellow")
        
        for task_name, result in results.items():
            status = "✓ Success" if result.success else "✗ Failed"
            status_style = "green" if result.success else "red"
            
            table.add_row(
                task_name,
                f"[{status_style}]{status}[/{status_style}]",
                f"{result.duration:.2f}s",
                str(result.exit_code)
            )
            
        self.console.print(table)
        
    def print_checkpoint_info(self, checkpoint_info: dict):
        """Print checkpoint information."""
        table = Table(title="Checkpoint Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in checkpoint_info.items():
            if key != 'completed_tasks':  # Skip long list
                table.add_row(key, str(value))
                
        self.console.print(table)
        
    def print_dag_info(self, dag_info: dict):
        """Print DAG information."""
        table = Table(title="DAG Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in dag_info.items():
            table.add_row(key, str(value))
            
        self.console.print(table)
        
    def log_task_output(self, task_name: str, output: str, is_error: bool = False):
        """Log task output to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_{task_name}_{timestamp}.log"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Task: {task_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Type: {'ERROR' if is_error else 'OUTPUT'}\n")
            f.write("-" * 50 + "\n")
            f.write(output)
            
        self.debug(f"Task output saved to: {filepath}")
        
    def create_progress_tracker(self, total_tasks: int):
        """Create a progress tracker for workflow execution."""
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        
        return progress
        
    def log_configuration(self, config: dict):
        """Log configuration information."""
        self.info("LightFlow Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
            
    def log_dependency_graph(self, dependencies: dict):
        """Log dependency graph information."""
        self.info("Dependency Graph:")
        for task, deps in dependencies.items():
            if deps:
                self.info(f"  {task} depends on: {', '.join(deps)}")
            else:
                self.info(f"  {task} (no dependencies)")
                
    def log_execution_plan(self, execution_order: list):
        """Log execution plan."""
        self.info("Execution Plan:")
        for i, batch in enumerate(execution_order):
            self.info(f"  Batch {i+1}: {', '.join(batch)}")
            
    def log_performance_metrics(self, metrics: dict):
        """Log performance metrics."""
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            table.add_row(metric, str(value))
            
        self.console.print(table) 
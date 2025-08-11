"""
Main CLI interface for Chroot Lite.
"""

import argparse
import sys
import os
import logging
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Add parent directory to path to import sandbox module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sandbox.manager import SandboxManager
from sandbox.executor import SandboxExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sandbox_history.log'),
        logging.StreamHandler()
    ]
)

console = Console()


class ChrootLiteCLI:
    """Command-line interface for Chroot Lite."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.manager = SandboxManager()
        self.console = Console()
        
    def create_sandbox(self, name: str, memory: int = 128, cpu: int = 30, 
                      allow_network: bool = False) -> None:
        """
        Create a new sandbox.
        
        Args:
            name: Name of the sandbox
            memory: Memory limit in MB
            cpu: CPU time limit in seconds
            allow_network: Whether to allow network access
        """
        try:
            success = self.manager.create_sandbox(
                name=name,
                memory_limit_mb=memory,
                cpu_limit_seconds=cpu,
                block_network=not allow_network
            )
            
            if success:
                self.console.print(f"✅ Created sandbox '{name}'", style="green")
                self.console.print(f"   Memory limit: {memory}MB")
                self.console.print(f"   CPU limit: {cpu}s")
                self.console.print(f"   Network: {'Allowed' if allow_network else 'Blocked'}")
            else:
                self.console.print(f"❌ Failed to create sandbox '{name}'", style="red")
                
        except Exception as e:
            self.console.print(f"❌ Error creating sandbox: {e}", style="red")
    
    def list_sandboxes(self) -> None:
        """List all sandboxes."""
        sandboxes = self.manager.list_sandboxes()
        
        if not sandboxes:
            self.console.print("No sandboxes found.", style="yellow")
            return
        
        table = Table(title="Sandboxes")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Memory Limit", style="blue")
        table.add_column("CPU Limit", style="blue")
        table.add_column("Network", style="magenta")
        table.add_column("Created", style="dim")
        
        for sandbox in sandboxes:
            status = self.manager.get_sandbox_status(sandbox['name'])
            network_status = "Allowed" if not sandbox.get('block_network', True) else "Blocked"
            
            table.add_row(
                sandbox['name'],
                status,
                f"{sandbox.get('memory_limit_mb', 128)}MB",
                f"{sandbox.get('cpu_limit_seconds', 30)}s",
                network_status,
                sandbox.get('created_at', 'Unknown')
            )
        
        self.console.print(table)
    
    def run_script(self, sandbox_name: str, script_path: str, args: Optional[list] = None) -> None:
        """
        Run a script in a sandbox.
        
        Args:
            sandbox_name: Name of the sandbox
            script_path: Path to the script
            args: Additional arguments
        """
        try:
            executor = self.manager.get_sandbox(sandbox_name)
            if not executor:
                self.console.print(f"❌ Sandbox '{sandbox_name}' not found", style="red")
                return
            
            self.console.print(f"🚀 Running script in sandbox '{sandbox_name}'...", style="blue")
            
            result = executor.execute_script(script_path, args or [])
            
            if result['success']:
                self.console.print("✅ Script executed successfully", style="green")
            else:
                self.console.print("❌ Script execution failed", style="red")
            
            self.console.print(f"Return code: {result['return_code']}")
            self.console.print(f"Execution time: {result['execution_time']:.2f}s")
            
            if result.get('stdout'):
                self.console.print("Output:", style="cyan")
                self.console.print(result['stdout'])
            
            if result.get('stderr'):
                self.console.print("Errors:", style="red")
                self.console.print(result['stderr'])
                
        except Exception as e:
            self.console.print(f"❌ Error running script: {e}", style="red")
    
    def run_python_code(self, sandbox_name: str, code: str) -> None:
        """
        Run Python code in a sandbox.
        
        Args:
            sandbox_name: Name of the sandbox
            code: Python code to execute
        """
        try:
            executor = self.manager.get_sandbox(sandbox_name)
            if not executor:
                self.console.print(f"❌ Sandbox '{sandbox_name}' not found", style="red")
                return
            
            self.console.print(f"🚀 Running Python code in sandbox '{sandbox_name}'...", style="blue")
            
            result = executor.execute_python_code(code)
            
            if result['success']:
                self.console.print("✅ Code executed successfully", style="green")
            else:
                self.console.print("❌ Code execution failed", style="red")
            
            self.console.print(f"Return code: {result['return_code']}")
            self.console.print(f"Execution time: {result['execution_time']:.2f}s")
            
            if result.get('stdout'):
                self.console.print("Output:", style="cyan")
                self.console.print(result['stdout'])
            
            if result.get('stderr'):
                self.console.print("Errors:", style="red")
                self.console.print(result['stderr'])
                
        except Exception as e:
            self.console.print(f"❌ Error running Python code: {e}", style="red")
    
    def delete_sandbox(self, name: str) -> None:
        """
        Delete a sandbox.
        
        Args:
            name: Name of the sandbox to delete
        """
        try:
            success = self.manager.delete_sandbox(name)
            
            if success:
                self.console.print(f"✅ Deleted sandbox '{name}'", style="green")
            else:
                self.console.print(f"❌ Failed to delete sandbox '{name}'", style="red")
                
        except Exception as e:
            self.console.print(f"❌ Error deleting sandbox: {e}", style="red")
    
    def show_sandbox_info(self, name: str) -> None:
        """
        Show detailed information about a sandbox.
        
        Args:
            name: Name of the sandbox
        """
        info = self.manager.get_sandbox_info(name)
        
        if not info:
            self.console.print(f"❌ Sandbox '{name}' not found", style="red")
            return
        
        panel = Panel(
            f"[cyan]Name:[/cyan] {info['name']}\n"
            f"[cyan]Path:[/cyan] {info['path']}\n"
            f"[cyan]Status:[/cyan] {self.manager.get_sandbox_status(name)}\n"
            f"[cyan]Memory Limit:[/cyan] {info.get('memory_limit_mb', 128)}MB\n"
            f"[cyan]CPU Limit:[/cyan] {info.get('cpu_limit_seconds', 30)}s\n"
            f"[cyan]Network:[/cyan] {'Allowed' if not info.get('block_network', True) else 'Blocked'}\n"
            f"[cyan]Created:[/cyan] {info.get('created_at', 'Unknown')}",
            title=f"Sandbox: {name}",
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def cleanup_all(self) -> None:
        """Clean up all sandboxes."""
        try:
            self.manager.cleanup_all()
            self.console.print("✅ Cleaned up all sandboxes", style="green")
        except Exception as e:
            self.console.print(f"❌ Error cleaning up sandboxes: {e}", style="red")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chroot Lite - A lightweight sandbox system for secure code execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create mybox --memory 256 --cpu 60
  %(prog)s run mybox script.py
  %(prog)s list
  %(prog)s delete mybox
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new sandbox')
    create_parser.add_argument('name', help='Name of the sandbox')
    create_parser.add_argument('--memory', '-m', type=int, default=128, 
                              help='Memory limit in MB (default: 128)')
    create_parser.add_argument('--cpu', '-c', type=int, default=30, 
                              help='CPU time limit in seconds (default: 30)')
    create_parser.add_argument('--allow-network', '-n', action='store_true',
                              help='Allow network access (default: blocked)')
    
    # List command
    subparsers.add_parser('list', help='List all sandboxes')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a script in a sandbox')
    run_parser.add_argument('sandbox', help='Name of the sandbox')
    run_parser.add_argument('script', help='Path to the script to run')
    run_parser.add_argument('args', nargs='*', help='Additional arguments for the script')
    
    # Python command
    python_parser = subparsers.add_parser('python', help='Run Python code in a sandbox')
    python_parser.add_argument('sandbox', help='Name of the sandbox')
    python_parser.add_argument('--code', '-c', required=True, help='Python code to execute')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a sandbox')
    delete_parser.add_argument('name', help='Name of the sandbox to delete')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show sandbox information')
    info_parser.add_argument('name', help='Name of the sandbox')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up all sandboxes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    cli = ChrootLiteCLI()
    
    try:
        if args.command == 'create':
            cli.create_sandbox(args.name, args.memory, args.cpu, args.allow_network)
        elif args.command == 'list':
            cli.list_sandboxes()
        elif args.command == 'run':
            cli.run_script(args.sandbox, args.script, args.args)
        elif args.command == 'python':
            cli.run_python_code(args.sandbox, args.code)
        elif args.command == 'delete':
            cli.delete_sandbox(args.name)
        elif args.command == 'info':
            cli.show_sandbox_info(args.name)
        elif args.command == 'cleanup':
            cli.cleanup_all()
            
    except KeyboardInterrupt:
        console.print("\n⚠️ Operation cancelled by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == '__main__':
    main() 
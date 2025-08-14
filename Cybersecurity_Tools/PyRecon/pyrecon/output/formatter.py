"""
Output formatting module for PyRecon.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from ..core.scanner import ScanResult


class OutputFormatter:
    """
    Formatter for PyRecon scan results with rich terminal output and JSON export.
    """
    
    def __init__(self, pretty: bool = True, json_output: Optional[str] = None):
        """
        Initialize the output formatter.
        
        Args:
            pretty: Whether to use pretty terminal output
            json_output: Path to save JSON output (optional)
        """
        self.pretty = pretty
        self.json_output = json_output
        self.console = Console()
        
        # Color schemes
        self.colors = {
            'open': 'green',
            'closed': 'red',
            'filtered': 'yellow',
            'tcp': 'blue',
            'udp': 'cyan',
            'http': 'magenta',
            'https': 'bright_magenta',
            'ssh': 'bright_green',
            'ftp': 'bright_blue',
            'smtp': 'bright_yellow',
            'dns': 'bright_cyan'
        }
    
    def format_results(self, results: List[ScanResult], target: str, 
                      scan_time: float = 0.0) -> None:
        """
        Format and display scan results.
        
        Args:
            results: List of scan results
            target: Target host
            scan_time: Time taken for scan
        """
        if self.pretty:
            self._print_pretty_results(results, target, scan_time)
        else:
            self._print_simple_results(results, target, scan_time)
        
        # Save JSON if requested
        if self.json_output:
            self._save_json_results(results, target, scan_time)
    
    def _print_pretty_results(self, results: List[ScanResult], target: str, 
                             scan_time: float) -> None:
        """
        Print results with rich formatting.
        
        Args:
            results: List of scan results
            target: Target host
            scan_time: Time taken for scan
        """
        # Header
        header = Text(f"🔍 PyRecon Scan Results", style="bold blue")
        self.console.print(Panel(header, box=box.ROUNDED))
        
        # Summary
        summary = self._create_summary_panel(results, target, scan_time)
        self.console.print(summary)
        
        if results:
            # Results table
            table = self._create_results_table(results)
            self.console.print(table)
            
            # Statistics
            stats = self._create_statistics_panel(results)
            self.console.print(stats)
        else:
            # No results
            no_results = Panel(
                Text("No open ports found", style="yellow"),
                title="Results",
                box=box.ROUNDED
            )
            self.console.print(no_results)
    
    def _print_simple_results(self, results: List[ScanResult], target: str, 
                            scan_time: float) -> None:
        """
        Print results in simple format.
        
        Args:
            results: List of scan results
            target: Target host
            scan_time: Time taken for scan
        """
        print(f"\nPyRecon Scan Results")
        print(f"Target: {target}")
        print(f"Scan Time: {scan_time:.2f}s")
        print(f"Open Ports: {len(results)}")
        print("-" * 50)
        
        for result in results:
            print(f"{result.port}/{result.protocol} - {result.service}")
            if result.banner:
                print(f"  Banner: {result.banner}")
            if result.os_guess:
                print(f"  OS: {result.os_guess}")
    
    def _create_summary_panel(self, results: List[ScanResult], target: str, 
                            scan_time: float) -> Panel:
        """
        Create summary panel.
        
        Args:
            results: List of scan results
            target: Target host
            scan_time: Time taken for scan
            
        Returns:
            Rich panel with summary
        """
        content = Text()
        content.append(f"Target: {target}\n", style="bold")
        content.append(f"Scan Time: {scan_time:.2f}s\n", style="cyan")
        content.append(f"Open Ports: {len(results)}", style="green")
        
        return Panel(content, title="Summary", box=box.ROUNDED)
    
    def _create_results_table(self, results: List[ScanResult]) -> Table:
        """
        Create results table.
        
        Args:
            results: List of scan results
            
        Returns:
            Rich table with results
        """
        table = Table(title="Open Ports", box=box.ROUNDED)
        
        # Add columns
        table.add_column("Port", style="cyan", no_wrap=True)
        table.add_column("Protocol", style="blue")
        table.add_column("Service", style="green")
        table.add_column("Banner", style="white")
        table.add_column("OS Guess", style="yellow")
        
        # Add rows
        for result in results:
            banner = result.banner[:50] + "..." if result.banner and len(result.banner) > 50 else result.banner
            os_guess = result.os_guess[:30] + "..." if result.os_guess and len(result.os_guess) > 30 else result.os_guess
            
            table.add_row(
                str(result.port),
                result.protocol.upper(),
                result.service,
                banner or "",
                os_guess or ""
            )
        
        return table
    
    def _create_statistics_panel(self, results: List[ScanResult]) -> Panel:
        """
        Create statistics panel.
        
        Args:
            results: List of scan results
            
        Returns:
            Rich panel with statistics
        """
        # Count services
        services = {}
        protocols = {}
        
        for result in results:
            # Service count
            service = result.service
            services[service] = services.get(service, 0) + 1
            
            # Protocol count
            protocol = result.protocol
            protocols[protocol] = protocols.get(protocol, 0) + 1
        
        # Create content
        content = Text()
        content.append("Service Distribution:\n", style="bold")
        
        for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True):
            content.append(f"  {service}: {count}\n", style="green")
        
        content.append("\nProtocol Distribution:\n", style="bold")
        for protocol, count in protocols.items():
            content.append(f"  {protocol.upper()}: {count}\n", style="blue")
        
        return Panel(content, title="Statistics", box=box.ROUNDED)
    
    def _save_json_results(self, results: List[ScanResult], target: str, 
                          scan_time: float) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: List of scan results
            target: Target host
            scan_time: Time taken for scan
        """
        try:
            # Convert results to JSON-serializable format
            json_results = {
                "target": target,
                "timestamp": datetime.now().isoformat(),
                "scan_time": scan_time,
                "open_ports": []
            }
            
            for result in results:
                port_info = {
                    "port": result.port,
                    "protocol": result.protocol,
                    "service": result.service,
                    "status": result.status
                }
                
                if result.banner:
                    port_info["banner"] = result.banner
                
                if result.os_guess:
                    port_info["os_guess"] = result.os_guess
                
                if result.tls_info:
                    port_info["tls_info"] = result.tls_info
                
                if result.response_time:
                    port_info["response_time"] = result.response_time
                
                json_results["open_ports"].append(port_info)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.json_output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save to file
            with open(self.json_output, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.console.print(f"\n[green]Results saved to: {self.json_output}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error saving JSON: {e}[/red]")
    
    def print_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Print progress information.
        
        Args:
            current: Current progress
            total: Total items
            message: Progress message
        """
        if self.pretty:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(message or "Scanning...", total=total)
                progress.update(task, completed=current)
        else:
            if message:
                print(f"{message}: {current}/{total}")
    
    def print_error(self, message: str) -> None:
        """
        Print error message.
        
        Args:
            message: Error message
        """
        if self.pretty:
            self.console.print(f"[red]Error: {message}[/red]")
        else:
            print(f"Error: {message}")
    
    def print_warning(self, message: str) -> None:
        """
        Print warning message.
        
        Args:
            message: Warning message
        """
        if self.pretty:
            self.console.print(f"[yellow]Warning: {message}[/yellow]")
        else:
            print(f"Warning: {message}")
    
    def print_success(self, message: str) -> None:
        """
        Print success message.
        
        Args:
            message: Success message
        """
        if self.pretty:
            self.console.print(f"[green]Success: {message}[/green]")
        else:
            print(f"Success: {message}")
    
    def format_scan_result(self, result: ScanResult) -> str:
        """
        Format a single scan result.
        
        Args:
            result: Scan result
            
        Returns:
            Formatted string
        """
        if self.pretty:
            return self._format_pretty_result(result)
        else:
            return self._format_simple_result(result)
    
    def _format_pretty_result(self, result: ScanResult) -> str:
        """
        Format result with rich styling.
        
        Args:
            result: Scan result
            
        Returns:
            Formatted string
        """
        text = Text()
        text.append(f"{result.port}/{result.protocol.upper()} ", style="cyan")
        text.append(f"{result.service} ", style="green")
        
        if result.banner:
            text.append(f"({result.banner[:30]}...) ", style="white")
        
        if result.os_guess:
            text.append(f"[OS: {result.os_guess}]", style="yellow")
        
        return text
    
    def _format_simple_result(self, result: ScanResult) -> str:
        """
        Format result in simple text.
        
        Args:
            result: Scan result
            
        Returns:
            Formatted string
        """
        output = f"{result.port}/{result.protocol.upper()} - {result.service}"
        
        if result.banner:
            output += f" ({result.banner[:50]})"
        
        if result.os_guess:
            output += f" [OS: {result.os_guess}]"
        
        return output 
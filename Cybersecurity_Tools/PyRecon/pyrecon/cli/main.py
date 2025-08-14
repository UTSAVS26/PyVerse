"""
Main CLI entry point for PyRecon.
"""

import time
import sys
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..core.scanner import PortScanner
from ..output.formatter import OutputFormatter


@click.group()
@click.version_option(version="1.0.0", prog_name="PyRecon")
def cli():
    """
    🔍 PyRecon: High-Speed Port Scanner & Service Fingerprinter
    
    A fast, multithreaded Python-based TCP/UDP port scanner with intelligent 
    service and OS fingerprinting capabilities.
    """
    pass


@cli.command()
@click.argument('target')
@click.option('-p', '--ports', default='top-100', 
              help='Port specification (e.g., 80,443, 1-1024, top-100)')
@click.option('--protocol', default='tcp', type=click.Choice(['tcp', 'udp']),
              help='Protocol to use (tcp/udp)')
@click.option('--fingerprint', is_flag=True, default=False,
              help='Perform service fingerprinting')
@click.option('--pretty', is_flag=True, default=True,
              help='Use pretty terminal output')
@click.option('--json', 'json_output', type=click.Path(),
              help='Save results to JSON file')
@click.option('--workers', default=100, type=int,
              help='Maximum number of worker threads')
@click.option('--timeout', default=1.0, type=float,
              help='Connection timeout in seconds')
@click.option('--file', '-f', type=click.Path(exists=True),
              help='Read targets from file')
def scan(target, ports, protocol, fingerprint, pretty, json_output, 
         workers, timeout, file):
    """
    Scan ports on target host(s).
    
    TARGET can be:
    - IP address (e.g., 192.168.1.1)
    - Domain name (e.g., example.com)
    - CIDR range (e.g., 192.168.1.0/24)
    - File path with targets (if --file is used)
    """
    console = Console()
    
    try:
        # Initialize scanner and formatter
        scanner = PortScanner(max_workers=workers, timeout=timeout)
        formatter = OutputFormatter(pretty=pretty, json_output=json_output)
        
        # Display scan configuration
        if pretty:
            config_panel = Panel(
                Text(f"Target: {target}\n"
                     f"Ports: {ports}\n"
                     f"Protocol: {protocol.upper()}\n"
                     f"Fingerprinting: {'Yes' if fingerprint else 'No'}\n"
                     f"Workers: {workers}\n"
                     f"Timeout: {timeout}s"),
                title="Scan Configuration",
                box=box.ROUNDED
            )
            console.print(config_panel)
        
        # Start scan
        start_time = time.time()
        
        # Progress callback
        def progress_callback(current, total):
            if pretty:
                formatter.print_progress(current, total, "Scanning hosts")
        
        # Perform scan
        results = scanner.scan(
            target=target,
            ports=ports,
            protocol=protocol,
            fingerprint=fingerprint,
            progress_callback=progress_callback
        )
        
        scan_time = time.time() - start_time
        
        # Display results
        formatter.format_results(results, target, scan_time)
        
        # Display statistics
        if results and pretty:
            stats = scanner.get_statistics(results)
            if stats:
                stats_text = Text()
                stats_text.append(f"Total Ports: {stats['total_ports']}\n", style="green")
                stats_text.append(f"Unique Hosts: {stats['unique_hosts']}\n", style="cyan")
                
                if stats['services']:
                    stats_text.append("Top Services:\n", style="bold")
                    for service, count in sorted(stats['services'].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]:
                        stats_text.append(f"  {service}: {count}\n", style="yellow")
                
                stats_panel = Panel(stats_text, title="Scan Statistics", box=box.ROUNDED)
                console.print(stats_panel)
        
    except Exception as e:
        if pretty:
            console.print(f"[red]Error: {str(e)}[/red]")
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('target')
@click.option('-p', '--ports', default='top-100',
              help='Port specification')
@click.option('--pretty', is_flag=True, default=True,
              help='Use pretty terminal output')
def quick(target, ports, pretty):
    """
    Perform a quick scan without fingerprinting.
    """
    scan.callback(target, ports, 'tcp', False, pretty, None, 100, 1.0, None)


@cli.command()
@click.argument('target')
@click.option('-p', '--ports', default='1-1024',
              help='Port specification')
@click.option('--pretty', is_flag=True, default=True,
              help='Use pretty terminal output')
@click.option('--json', 'json_output', type=click.Path(),
              help='Save results to JSON file')
def full(target, ports, pretty, json_output):
    """
    Perform a full scan with fingerprinting.
    """
    scan.callback(target, ports, 'tcp', True, pretty, json_output, 100, 1.0, None)


@cli.command()
@click.argument('target')
@click.option('-p', '--ports', default='top-100',
              help='Port specification')
@click.option('--pretty', is_flag=True, default=True,
              help='Use pretty terminal output')
def udp(target, ports, pretty):
    """
    Perform UDP port scan.
    """
    scan.callback(target, ports, 'udp', True, pretty, None, 100, 1.0, None)


@cli.command()
def version():
    """
    Show version information.
    """
    console = Console()
    
    version_text = Text()
    version_text.append("PyRecon v1.0.0\n", style="bold blue")
    version_text.append("High-Speed Port Scanner & Service Fingerprinter\n", style="cyan")
    version_text.append("Author: Shivansh Katiyar\n", style="green")
    version_text.append("SSOC Participant", style="yellow")
    
    version_panel = Panel(version_text, box=box.ROUNDED)
    console.print(version_panel)


@cli.command()
def help():
    """
    Show detailed help information.
    """
    console = Console()
    
    help_text = Text()
    help_text.append("PyRecon Usage Examples:\n\n", style="bold blue")
    
    help_text.append("Quick scan of common ports:\n", style="green")
    help_text.append("  pyrecon scan 192.168.1.1 --top-ports 100\n\n", style="white")
    
    help_text.append("Full scan with fingerprinting:\n", style="green")
    help_text.append("  pyrecon scan example.com -p 1-1024 --fingerprint\n\n", style="white")
    
    help_text.append("UDP scan:\n", style="green")
    help_text.append("  pyrecon scan 10.0.0.1 --protocol udp\n\n", style="white")
    
    help_text.append("Save results to JSON:\n", style="green")
    help_text.append("  pyrecon scan target.com --json results.json\n\n", style="white")
    
    help_text.append("Scan multiple targets from file:\n", style="green")
    help_text.append("  pyrecon scan -f targets.txt --fingerprint\n\n", style="white")
    
    help_text.append("Port specifications:\n", style="yellow")
    help_text.append("  - Single port: 80\n", style="white")
    help_text.append("  - Port range: 1-1024\n", style="white")
    help_text.append("  - Port list: 80,443,8080\n", style="white")
    help_text.append("  - Top ports: top-100\n", style="white")
    
    help_panel = Panel(help_text, title="Help", box=box.ROUNDED)
    console.print(help_panel)


def main():
    """
    Main entry point for the CLI.
    """
    cli()


if __name__ == '__main__':
    main() 
"""
CLI interface for ClipCrypt.

Command-line interface using Click for the ClipCrypt clipboard manager.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from .core import ClipCrypt


@click.group()
@click.version_option(version="1.0.0")
@click.option('--config-dir', type=click.Path(), help='Custom configuration directory')
@click.pass_context
def cli(ctx, config_dir):
    """🔐 ClipCrypt: Encrypted Clipboard Manager
    
    Secure, searchable, and local-only clipboard history — all encrypted.
    """
    ctx.ensure_object(dict)
    
    if config_dir:
        ctx.obj['config_dir'] = Path(config_dir)
    else:
        ctx.obj['config_dir'] = None


@cli.command()
@click.pass_context
def monitor(ctx):
    """Start monitoring clipboard for changes."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    
    console = Console()
    console.print(Panel(
        "🕵️  Starting clipboard monitoring...\n"
        "Press Ctrl+C to stop monitoring",
        title="ClipCrypt Monitor",
        border_style="green"
    ))
    
    try:
        clipcrypt.start_monitoring()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user.[/yellow]")


@cli.command()
@click.option('--limit', '-l', type=int, help='Maximum number of entries to show')
@click.pass_context
def list(ctx, limit):
    """List all clipboard entries."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.list_entries(limit)


@cli.command()
@click.argument('entry_id', type=int)
@click.pass_context
def get(ctx, entry_id):
    """Get a specific clipboard entry."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.get_entry(entry_id)


@cli.command()
@click.argument('query')
@click.pass_context
def search(ctx, query):
    """Search clipboard entries by content."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.search_entries(query)


@cli.command()
@click.argument('entry_id', type=int)
@click.pass_context
def delete(ctx, entry_id):
    """Delete a clipboard entry."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.delete_entry(entry_id)


@cli.command()
@click.argument('entry_id', type=int)
@click.argument('tag')
@click.pass_context
def tag(ctx, entry_id, tag):
    """Add a tag to a clipboard entry."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.add_tag(entry_id, tag)


@cli.command()
@click.argument('entry_id', type=int)
@click.argument('tag')
@click.pass_context
def untag(ctx, entry_id, tag):
    """Remove a tag from a clipboard entry."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.remove_tag(entry_id, tag)


@cli.command()
@click.argument('tag')
@click.pass_context
def bytag(ctx, tag):
    """List all entries with a specific tag."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.get_entries_by_tag(tag)


@cli.command()
@click.pass_context
def clear(ctx):
    """Clear all clipboard entries."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.clear_all()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show storage statistics."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.show_stats()


@cli.command()
@click.argument('entry_id', type=int)
@click.pass_context
def copy(ctx, entry_id):
    """Copy an entry back to the clipboard."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    clipcrypt.copy_to_clipboard(entry_id)


@cli.command()
@click.pass_context
def info(ctx):
    """Show ClipCrypt information and configuration."""
    clipcrypt = ClipCrypt(ctx.obj['config_dir'])
    
    console = Console()
    
    info_text = f"""
🔐 ClipCrypt v1.0.0
Author: Shivansh Katiyar

Configuration Directory: {clipcrypt.config_dir}
Data File: {clipcrypt.storage_manager.data_file}
Key File: {clipcrypt.storage_manager.encryption_manager.key_file}

Features:
• AES-GCM encryption for all clipboard data
• Local-only storage (no cloud dependencies)
• Search and tag functionality
• Cross-platform clipboard monitoring
• Rich CLI interface with colored output
    """.strip()
    
    panel = Panel(
        info_text,
        title="ClipCrypt Information",
        border_style="cyan"
    )
    console.print(panel)


if __name__ == '__main__':
    cli() 
#!/usr/bin/env python3
"""
Demo script for PyTerminus.
This script demonstrates the basic functionality of PyTerminus.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def demo_session_manager():
    """Demo the session manager functionality."""
    print("🖥️  PyTerminus Demo")
    print("=" * 50)
    
    from core.session_manager import SessionManager
    from core.logger import Logger
    
    # Create a temporary log directory
    log_dir = Path("./demo_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Initialize components
    logger = Logger(log_dir)
    session_manager = SessionManager(
        default_shell='bash',
        log_dir=log_dir,
        logger=logger
    )
    
    print("✅ Session Manager initialized")
    
    # Demo session info
    info = session_manager.get_session_info()
    print(f"📊 Session Info: {info}")
    
    # Demo logging
    logger.log_session_started("demo-session", 0)
    print("📝 Logged session start")
    
    # Demo session summary
    summary = logger.get_session_summary()
    print(f"📈 Session Summary: {summary}")
    
    print("\n✅ Demo completed successfully!")
    print("💡 Note: Terminal panes are not supported on Windows")
    print("   The core functionality (session management, logging) works on all platforms")

def demo_keybindings():
    """Demo the keybindings functionality."""
    print("\n⌨️  Keybindings Demo")
    print("=" * 30)
    
    from tui.keybindings import KeyBindings
    
    kb = KeyBindings()
    
    # Test some bindings
    print(f"Ctrl+n → {kb.get_binding('ctrl n')}")
    print(f"Ctrl+q → {kb.get_binding('ctrl q')}")
    print(f"F1 → {kb.get_binding('f1')}")
    
    # Show help text
    help_text = kb.get_help_text()
    print(f"\n📖 Help text length: {len(help_text)} characters")
    print("✅ Keybindings demo completed!")

def demo_layout():
    """Demo the layout functionality."""
    print("\n🪟 Layout Demo")
    print("=" * 20)
    
    from tui.layout import LayoutManager
    import urwid
    
    layout = LayoutManager()
    
    # Create some mock widgets
    widget1 = urwid.Text("Pane 1")
    widget2 = urwid.Text("Pane 2")
    
    # Add panes
    layout.add_pane("pane1", widget1)
    layout.add_pane("pane2", widget2)
    
    print(f"📊 Pane count: {len(layout.panes)}")
    print(f"🎯 Active pane: {layout.get_active_pane()}")
    print(f"📋 Pane names: {layout.get_pane_names()}")
    
    # Test status bar
    status = layout.get_status_bar_text()
    print(f"📊 Status: {status}")
    
    print("✅ Layout demo completed!")

def demo_logger():
    """Demo the logger functionality."""
    print("\n📝 Logger Demo")
    print("=" * 20)
    
    from core.logger import Logger
    
    # Create logger
    log_dir = Path("./demo_logs")
    logger = Logger(log_dir)
    
    # Log some events
    logger.log_session_started("demo", 2)
    logger.log_pane_created("pane1", "bash", "/home")
    logger.log_pane_created("pane2", "zsh", "/tmp")
    logger.log_command_executed("pane1", "ls -la")
    logger.log_pane_output("pane1", "Hello, World!\n")
    
    # Get recent logs
    logs = logger.get_recent_logs('session', limit=5)
    print(f"📊 Recent session logs: {len(logs)} entries")
    
    # Search logs
    results = logger.search_logs("pane", 'session')
    print(f"🔍 Search results for 'pane': {len(results)} entries")
    
    # Get summary
    summary = logger.get_session_summary()
    print(f"📈 Summary: {summary}")
    
    print("✅ Logger demo completed!")

def main():
    """Run all demos."""
    print("🚀 Starting PyTerminus Demo")
    print("=" * 50)
    
    try:
        demo_session_manager()
        demo_keybindings()
        demo_layout()
        demo_logger()
        
        print("\n🎉 All demos completed successfully!")
        print("=" * 50)
        print("📁 Check the 'demo_logs' directory for generated log files")
        print("💡 The application is ready for use on supported platforms")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
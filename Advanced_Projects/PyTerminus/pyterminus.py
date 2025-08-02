#!/usr/bin/env python3
"""
PyTerminus: Virtual Multi-Terminal Manager in Python
A powerful, Python-based terminal session manager — split, persist, search.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.session_manager import SessionManager
from tui.main_ui import MainUI
from core.logger import Logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTerminus - Virtual Multi-Terminal Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyterminus                           # Start with default session
  pyterminus --session dev-work.yaml   # Load saved session profile
  pyterminus --log-dir ./logs/         # Specify custom log directory
  pyterminus --theme dark              # Use dark theme
  pyterminus --shell zsh               # Use zsh as default shell
        """
    )
    
    parser.add_argument(
        '--session', 
        type=str, 
        help='Load saved session profile (YAML file)'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./logs',
        help='Specify custom log directory (default: ./logs)'
    )
    parser.add_argument(
        '--theme', 
        choices=['dark', 'light'], 
        default='dark',
        help='UI theme (default: dark)'
    )
    parser.add_argument(
        '--shell', 
        type=str, 
        default='bash',
        help='Default shell for new panes (default: bash)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for PyTerminus."""
    args = parse_arguments()
    
    # Ensure log directory exists
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_dir)
    
    try:
        # Initialize session manager
        session_manager = SessionManager(
            default_shell=args.shell,
            log_dir=log_dir,
            logger=logger
        )
        
        # Load session if specified
        if args.session:
            session_path = Path(args.session)
            if session_path.exists():
                session_manager.load_session(session_path)
                print(f"✔ Loaded session: {session_manager.get_session_info()}")
            else:
                print(f"⚠ Warning: Session file '{args.session}' not found")
        
        # Initialize and run UI
        ui = MainUI(
            session_manager=session_manager,
            theme=args.theme,
            logger=logger
        )
        
        ui.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        if args.debug:
            raise
        else:
            print(f"❌ Error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main() 
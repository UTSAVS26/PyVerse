#!/usr/bin/env python3
"""
Main entry point for the Accent Strength Estimator application.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.cli_interface import CLIInterface
from src.ui.gui_interface import GUIInterface
from src.ui.web_interface import WebInterface


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Accent Strength Estimator - Analyze English pronunciation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode cli          # Run command-line interface
  python main.py --mode gui          # Run GUI interface
  python main.py --mode web          # Run web interface
  python main.py --help              # Show this help message
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['cli', 'gui', 'web'],
        default='cli',
        help='Interface mode to use (default: cli)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'cli':
            interface = CLIInterface()
            interface.run()
        elif args.mode == 'gui':
            interface = GUIInterface()
            interface.run()
        elif args.mode == 'web':
            interface = WebInterface()
            interface.run()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def show_detailed_help():
    """Show detailed help information."""
    help_text = """
🎤 Accent Strength Estimator

A comprehensive tool for analyzing English pronunciation and estimating accent strength.

FEATURES:
    • Real-time speech recording and analysis
    • Phoneme-level pronunciation comparison
    • Pitch contour and intonation analysis
    • Speech rhythm and timing analysis
    • Personalized feedback and improvement tips
    • Multiple interface options (CLI, GUI, Web)

INTERFACE MODES:

CLI Mode (Command Line):
    • Text-based interface
    • Step-by-step guidance
    • Detailed results display
    • Suitable for all environments

GUI Mode (Graphical):
    • Desktop application interface
    • Visual recording controls
    • Real-time feedback display
    • Interactive results visualization

Web Mode (Browser):
    • Web-based interface
    • Accessible from any device
    • Modern responsive design
    • Cloud-based processing (if configured)

USAGE EXAMPLES:

1. Command Line Interface:
   python main.py --mode cli

2. Graphical Interface:
   python main.py --mode gui

3. Web Interface:
   python main.py --mode web

TIPS FOR BEST RESULTS:

• Audio Quality:
  - Use a good quality microphone
  - Minimize background noise
  - Ensure stable internet connection (for web mode)

• Speaking:
  - Speak clearly and at normal pace
  - Practice phrases before recording
  - Maintain consistent volume

• Environment:
  - Choose a quiet location
  - Avoid echo and reverberation
  - Ensure proper microphone positioning

TECHNICAL REQUIREMENTS:

• Python 3.7 or higher
• Required packages (see requirements.txt)
• Microphone access
• Sufficient disk space for audio files

For more information, visit the project documentation.
    """
    print(help_text)


if __name__ == "__main__":
    main()

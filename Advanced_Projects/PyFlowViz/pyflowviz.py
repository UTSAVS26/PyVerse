#!/usr/bin/env python3
"""
PyFlowViz: Code-to-Flowchart Auto Generator
CLI entry point
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import ASTParser, BytecodeParser
from visualizer import GraphvizGenerator, HTMLRenderer

# Optional GUI imports
try:
    from gui import GradioGUI, PyQt5GUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyFlowViz: Code-to-Flowchart Auto Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyflowviz example.py --output flow.svg
  pyflowviz app.py --html
  pyflowviz --batch src/ --output output/
  pyflowviz --gui
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Python file to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (SVG/PNG/HTML)'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate interactive HTML output'
    )
    
    parser.add_argument(
        '--mermaid',
        action='store_true',
        help='Generate Mermaid.js HTML output'
    )
    
    parser.add_argument(
        '--batch',
        help='Process all Python files in directory'
    )
    
    parser.add_argument(
        '--parser',
        choices=['ast', 'bytecode'],
        default='ast',
        help='Parser type (default: ast)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface'
    )
    
    parser.add_argument(
        '--gradio',
        action='store_true',
        help='Launch Gradio web interface'
    )
    
    parser.add_argument(
        '--pyqt',
        action='store_true',
        help='Launch PyQt5 desktop interface'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle GUI modes
    if args.gui or args.gradio or args.pyqt:
        if not GUI_AVAILABLE:
            print("GUI dependencies not available. Install them with:")
            print("   pip install gradio PyQt5")
            return
        
        if args.gradio:
            print("Launching Gradio web interface...")
            gui = GradioGUI()
            gui.launch(share=False)
        elif args.pyqt:
            print("Launching PyQt5 desktop interface...")
            gui = PyQt5GUI()
            gui.launch()
        else:
            print("Launching Gradio web interface...")
            gui = GradioGUI()
            gui.launch(share=False)
        return
    
    # Handle batch processing
    if args.batch:
        process_batch(args.batch, args.output, args.parser, args.verbose)
        return
    
    # Handle single file processing
    if not args.file:
        parser.print_help()
        return
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return
    
    process_single_file(args.file, args.output, args.html, args.mermaid, args.parser, args.verbose)


def process_single_file(file_path: str, output_path: str = None, html: bool = False, 
                       mermaid: bool = False, parser_type: str = 'ast', verbose: bool = False):
    """Process a single Python file."""
    try:
        if verbose:
            print(f"Parsing file: {file_path}")
        
        # Select parser
        if parser_type == 'ast':
            parser = ASTParser()
        else:
            parser = BytecodeParser()
        
        # Parse file
        flowchart_data = parser.parse_file(file_path)
        
        if verbose:
            print(f"Parsed successfully: {len(flowchart_data['nodes'])} nodes, {len(flowchart_data['edges'])} edges")
        
        # Determine output format and path
        if html:
            output_format = "Interactive HTML"
            if not output_path:
                output_path = f"{os.path.splitext(file_path)[0]}_flowchart.html"
        elif mermaid:
            output_format = "Mermaid HTML"
            if not output_path:
                output_path = f"{os.path.splitext(file_path)[0]}_flowchart_mermaid.html"
        else:
            # Determine format from output path
            if output_path:
                ext = os.path.splitext(output_path)[1].lower()
                if ext == '.svg':
                    output_format = "SVG"
                elif ext == '.png':
                    output_format = "PNG"
                elif ext == '.html':
                    output_format = "Interactive HTML"
                else:
                    output_format = "SVG"
                    if not output_path.endswith('.svg'):
                        output_path += '.svg'
            else:
                output_format = "SVG"
                output_path = f"{os.path.splitext(file_path)[0]}_flowchart.svg"
        
        # Generate output
        if output_format == "SVG":
            generator = GraphvizGenerator()
            result_path = generator.generate_svg(flowchart_data, output_path)
        elif output_format == "PNG":
            generator = GraphvizGenerator()
            result_path = generator.generate_png(flowchart_data, output_path)
        elif output_format == "Interactive HTML":
            renderer = HTMLRenderer()
            result_path = renderer.generate_html(flowchart_data, output_path)
        else:  # Mermaid HTML
            renderer = HTMLRenderer()
            result_path = renderer.generate_mermaid_html(flowchart_data, output_path)
        
        print(f"Generated {output_format}: {result_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        sys.exit(1)


def process_batch(directory: str, output_dir: str = None, parser_type: str = 'ast', verbose: bool = False):
    """Process all Python files in a directory."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    if not python_files:
        print(f"No Python files found in '{directory}'")
        return
    
    print(f"Found {len(python_files)} Python files in '{directory}'")
    
    # Process each file
    success_count = 0
    for file_path in python_files:
        try:
            if verbose:
                print(f"\nProcessing: {file_path}")
            
            # Determine output path
            if output_dir:
                rel_path = os.path.relpath(file_path, directory)
                base_name = os.path.splitext(rel_path)[0]
                output_path = os.path.join(output_dir, f"{base_name}_flowchart.svg")
            else:
                output_path = f"{os.path.splitext(file_path)[0]}_flowchart.svg"
            
            process_single_file(file_path, output_path, parser_type=parser_type, verbose=verbose)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nSuccessfully processed {success_count}/{len(python_files)} files")


if __name__ == "__main__":
    main() 
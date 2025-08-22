#!/usr/bin/env python3
"""
Command Line Interface for RiddleMind.

This module provides a command-line interface for the RiddleMind logic puzzle solver.
"""

import argparse
import sys
import os
from pathlib import Path
from .solver import RiddleMind


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🧠 RiddleMind - Logic Puzzle Solver Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m riddlemind.cli

  # Solve a puzzle from command line
  python -m riddlemind.cli -p "Alice is older than Bob. Charlie is younger than Alice. Who is the oldest?"

  # Solve a puzzle from file
  python -m riddlemind.cli -f puzzle.txt

  # Verbose output
  python -m riddlemind.cli -p "Alice is older than Bob" -v
        """
    )
    
    parser.add_argument(
        "-p", "--puzzle",
        type=str,
        help="Logic puzzle text to solve"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="File containing the logic puzzle"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed reasoning steps"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="RiddleMind 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Initialize solver
    try:
        solver = RiddleMind()
    except Exception as e:
        print(f"❌ Error initializing RiddleMind: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    # Determine puzzle text
    puzzle_text = None
    
    if args.puzzle:
        puzzle_text = args.puzzle
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                puzzle_text = f.read().strip()
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.interactive or (not args.puzzle and not args.file):
        # Interactive mode
        puzzle_text = get_interactive_input()
    
    if not puzzle_text:
        print("❌ No puzzle provided. Use -p, -f, or --interactive")
        sys.exit(1)
    
    # Solve the puzzle
    try:
        print("🧠 RiddleMind is thinking...")
        solution = solver.solve(puzzle_text)
        
        # Display results
        if args.verbose:
            print(solution)
        else:
            print_simplified_solution(solution)
            
    except Exception as e:
        print(f"❌ Error solving puzzle: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def get_interactive_input() -> str:
    """Get puzzle input interactively."""
    print("🧠 Welcome to RiddleMind!")
    print("Enter your logic puzzle below.")
    print("Press Enter twice when you're done:")
    print("-" * 50)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "" and lines:
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            break
    
    return "\n".join(lines)


def print_simplified_solution(solution):
    """Print a simplified version of the solution."""
    print("\n" + "=" * 50)
    print("🧠 RIDDLEMIND SOLUTION")
    print("=" * 50)
    
    # Original puzzle
    print(f"\n📝 Puzzle:")
    print(f"   {solution.original_text}")
    
    # Parsed constraints
    if solution.parsed_constraints:
        print(f"\n🔍 Parsed {len(solution.parsed_constraints)} constraints:")
        for constraint in solution.parsed_constraints:
            print(f"   • {constraint}")
    
    # Questions
    if solution.questions:
        print(f"\n❓ Questions:")
        for question in solution.questions:
            print(f"   • {question}")
    
    # Conclusions
    if solution.conclusions:
        print(f"\n✅ Key Conclusions:")
        for conclusion in solution.conclusions:
            if not conclusion.startswith("Found") and not conclusion.startswith("Entities"):
                print(f"   • {conclusion}")
    
    # Warnings
    if solution.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in solution.warnings:
            print(f"   • {warning}")
    
    # Errors
    if solution.errors:
        print(f"\n❌ Errors:")
        for error in solution.errors:
            print(f"   • {error}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()

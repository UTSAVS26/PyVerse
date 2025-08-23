#!/usr/bin/env python3
"""
Example usage of RiddleMind API.

This script demonstrates how to use the RiddleMind logic puzzle solver
programmatically.
"""

from riddlemind import RiddleMind


def main():
    """Demonstrate RiddleMind functionality."""
    
    # Initialize the solver
    solver = RiddleMind()
    
    print("🧠 RiddleMind Example Usage")
    print("=" * 50)
    
    # Example 1: Simple age comparison
    print("\n📝 Example 1: Simple Age Comparison")
    puzzle1 = "Alice is older than Bob. Charlie is younger than Alice. Who is the oldest?"
    
    solution1 = solver.solve(puzzle1)
    print(f"Puzzle: {puzzle1}")
    print(f"Parsed constraints: {len(solution1.parsed_constraints)}")
    print(f"Questions: {len(solution1.questions)}")
    print(f"Conclusions: {len(solution1.conclusions)}")
    
    # Example 2: Complex transitive relationships
    print("\n📝 Example 2: Complex Transitive Relationships")
    puzzle2 = """
    Alice is older than Bob.
    Bob is taller than Charlie.
    Charlie is older than David.
    Alice is Bob's sister.
    Who is the oldest?
    Who is the tallest?
    """
    
    solution2 = solver.solve(puzzle2)
    print(f"Puzzle: {puzzle2.strip()}")
    print(f"Parsed constraints: {len(solution2.parsed_constraints)}")
    print(f"Questions: {len(solution2.questions)}")
    print(f"Conclusions: {len(solution2.conclusions)}")
    
    # Example 3: Spatial arrangement
    print("\n📝 Example 3: Spatial Arrangement")
    puzzle3 = "Alice sits to the left of Bob. Bob sits to the left of Charlie. Who is in the middle?"
    
    solution3 = solver.solve(puzzle3)
    print(f"Puzzle: {puzzle3}")
    print(f"Parsed constraints: {len(solution3.parsed_constraints)}")
    print(f"Questions: {len(solution3.questions)}")
    print(f"Conclusions: {len(solution3.conclusions)}")
    
    # Example 4: Contradiction detection
    print("\n📝 Example 4: Contradiction Detection")
    puzzle4 = "Alice is older than Bob. Bob is older than Alice."
    
    solution4 = solver.solve(puzzle4)
    print(f"Puzzle: {puzzle4}")
    print(f"Parsed constraints: {len(solution4.parsed_constraints)}")
    print(f"Errors: {len(solution4.errors)}")
    if solution4.errors:
        print(f"Error detected: {solution4.errors[0]}")
    
    # Example 5: Family relationships
    print("\n📝 Example 5: Family Relationships")
    puzzle5 = "Alice is Bob's sister. Bob is Charlie's brother. What is Alice's relationship to Charlie?"
    
    solution5 = solver.solve(puzzle5)
    print(f"Puzzle: {puzzle5}")
    print(f"Parsed constraints: {len(solution5.parsed_constraints)}")
    print(f"Questions: {len(solution5.questions)}")
    print(f"Conclusions: {len(solution5.conclusions)}")
    
    print("\n✅ All examples completed successfully!")
    print("\nTo run the web interface:")
    print("  streamlit run riddlemind/web_app.py")
    print("\nTo use the command line interface:")
    print("  python -m riddlemind.cli -p \"Your puzzle here\"")


if __name__ == "__main__":
    main()

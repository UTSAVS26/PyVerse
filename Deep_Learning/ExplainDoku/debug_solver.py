#!/usr/bin/env python3
"""
Debug script to check solver behavior
"""

from explaindoku.core.grid import Grid
from explaindoku.core.solver import Solver

def main():
    # Test the hard puzzle that's failing
    grid_str = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
    grid = Grid.from_string(grid_str)
    solver = Solver(grid)
    
    print("Initial grid:")
    print(grid)
    print(f"Initial empty cells: {len(grid.get_empty_cells())}")
    print(f"Initial is_solved: {grid.is_solved()}")
    print(f"Initial is_valid: {grid.is_valid()}")
    
    # Apply human strategies
    human_steps = solver._apply_human_strategies()
    print(f"\nApplied {len(human_steps)} human steps")
    
    print("\nFinal grid:")
    print(solver.cm.grid)
    print(f"Final empty cells: {len(solver.cm.grid.get_empty_cells())}")
    print(f"Final is_solved: {solver.cm.grid.is_solved()}")
    print(f"Final is_valid: {solver.cm.grid.is_valid()}")
    
    # Check if any cells have no candidates
    empty_cells = solver.cm.grid.get_empty_cells()
    for row, col in empty_cells:
        candidates = solver.cm.get_candidates(row, col)
        if not candidates:
            print(f"Cell ({row}, {col}) has no candidates!")
    
    # Try to solve
    result = solver.solve(use_search=True)
    print(f"\nSolve result:")
    print(f"Success: {result.success}")
    print(f"Total steps: {result.total_steps}")
    print(f"Human steps: {result.human_steps}")
    print(f"Search steps: {result.search_steps}")
    print(f"Final grid solved: {result.final_grid.is_solved()}")

if __name__ == "__main__":
    main()

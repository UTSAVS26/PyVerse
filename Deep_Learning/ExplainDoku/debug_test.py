#!/usr/bin/env python3
"""
Debug script to test solver issue
"""

from explaindoku.core.grid import Grid
from explaindoku.core.solver import Solver

# Test the puzzle that's failing
grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
grid = Grid.from_string(grid_str)
solver = Solver(grid)

print("Initial grid:")
print(grid.to_display_string())
print(f"Empty cells: {len(grid.get_empty_cells())}")

# Apply human strategies
result = solver.solve(use_search=False)

print(f"\nResult success: {result.success}")
print(f"Total steps: {result.total_steps}")
print(f"Human steps: {result.human_steps}")
print(f"Final grid solved: {result.final_grid.is_solved()}")
print(f"Final empty cells: {len(result.final_grid.get_empty_cells())}")

# Check if the grid is actually solved
if result.final_grid.is_solved():
    print("✅ Grid is actually solved!")
else:
    print("❌ Grid is not solved!")
    print("\nFinal grid:")
    print(result.final_grid.to_display_string())

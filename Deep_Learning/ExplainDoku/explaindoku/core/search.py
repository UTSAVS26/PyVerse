"""
Search strategies for Sudoku solving (backtracking with heuristics)
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .constraints import ConstraintManager
from .grid import Grid


@dataclass
class SearchResult:
    """Result of a search step"""
    cell_position: Tuple[int, int]
    value: int
    technique: str = "backtracking"
    eliminations: List[Tuple[Tuple[int, int], int]] = None
    backtrack_count: int = 0
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []


class SearchStrategy:
    """Implements search strategies for Sudoku solving"""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.cm = constraint_manager
        self.backtrack_count = 0
    
    def solve_with_backtracking(self, max_backtracks: int = 10000) -> Optional[Grid]:
        """Solve the puzzle using backtracking with heuristics"""
        self.backtrack_count = 0
        return self._backtrack(self.cm.grid.copy(), max_backtracks)
    
    def _backtrack(self, grid: Grid, max_backtracks: int) -> Optional[Grid]:
        """Recursive backtracking with heuristics"""
        if self.backtrack_count >= max_backtracks:
            return None
        
        # Check if solved
        if grid.is_solved():
            return grid
        
        # Check if invalid
        if not grid.is_valid():
            return None
        
        # Get next cell to try (MRV heuristic)
        next_cell = self._get_next_cell(grid)
        if next_cell is None:
            return None
        
        row, col = next_cell
        
        # Get values to try (LCV heuristic)
        values = self._get_values_to_try(grid, row, col)
        
        for value in values:
            # Try this value
            if grid.is_valid_move(row, col, value):
                # Make a copy and set the value
                new_grid = grid.copy()
                new_grid.set_value(row, col, value)
                
                # Recursive call
                result = self._backtrack(new_grid, max_backtracks)
                if result is not None:
                    return result
                
                self.backtrack_count += 1
        
        return None
    
    def _get_next_cell(self, grid: Grid) -> Optional[Tuple[int, int]]:
        """Get next cell to try using MRV (Minimum Remaining Values) heuristic"""
        empty_cells = grid.get_empty_cells()
        
        if not empty_cells:
            return None
        
        # Use MRV heuristic
        return min(empty_cells, key=lambda pos: self._get_candidate_count(grid, *pos))
    
    def _get_candidate_count(self, grid: Grid, row: int, col: int) -> int:
        """Get number of valid candidates for a cell"""
        count = 0
        for value in range(1, 10):
            if grid.is_valid_move(row, col, value):
                count += 1
        return count
    
    def _get_values_to_try(self, grid: Grid, row: int, col: int) -> List[int]:
        """Get values to try using LCV (Least Constraining Value) heuristic"""
        valid_values = []
        for value in range(1, 10):
            if grid.is_valid_move(row, col, value):
                valid_values.append(value)
        
        # Sort by LCV heuristic (least constraining first)
        return sorted(valid_values, key=lambda v: self._get_constraint_count(grid, row, col, v))
    
    def _get_constraint_count(self, grid: Grid, row: int, col: int, value: int) -> int:
        """Get number of constraints this value would impose"""
        constraints = 0
        
        # Check how many empty cells in peers would be affected
        for peer_row, peer_col in grid.get_peers(row, col):
            if grid.get_value(peer_row, peer_col) is None:
                # Check if this value is a candidate for the peer
                if grid.is_valid_move(peer_row, peer_col, value):
                    constraints += 1
        
        return constraints
    
    def get_search_explanation(self, result: SearchResult) -> str:
        """Generate human-readable explanation for a search result"""
        row, col = result.cell_position
        cell_pos = f"R{row+1}C{col+1}"
        
        if result.technique == "backtracking":
            return f"Backtracking: Try {result.value} in {cell_pos} (backtrack #{result.backtrack_count})."
        
        return f"Search: Place {result.value} in {cell_pos}."
    
    def solve_step_by_step(self, max_steps: int = 100) -> List[SearchResult]:
        """Solve step by step, returning each step"""
        steps = []
        grid = self.cm.grid.copy()
        
        for step in range(max_steps):
            if grid.is_solved():
                break
            
            if not grid.is_valid():
                break
            
            # Get next cell and value
            next_cell = self._get_next_cell(grid)
            if next_cell is None:
                break
            
            row, col = next_cell
            values = self._get_values_to_try(grid, row, col)
            
            if not values:
                break
            
            # Try the first value
            value = values[0]
            if grid.is_valid_move(row, col, value):
                grid.set_value(row, col, value)
                
                steps.append(SearchResult(
                    cell_position=(row, col),
                    value=value,
                    technique="backtracking",
                    backtrack_count=self.backtrack_count
                ))
            else:
                break
        
        return steps

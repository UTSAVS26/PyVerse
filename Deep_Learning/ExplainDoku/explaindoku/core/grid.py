"""
Grid representation and utilities for Sudoku boards
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass


@dataclass
class Cell:
    """Represents a single cell in the Sudoku grid"""
    row: int
    col: int
    value: Optional[int] = None
    candidates: Set[int] = None
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = set(range(1, 10)) if self.value is None else set()
    
    @property
    def position(self) -> str:
        """Return cell position in R1C1 format"""
        return f"R{self.row+1}C{self.col+1}"
    
    @property
    def is_filled(self) -> bool:
        """Check if cell has a value"""
        return self.value is not None
    
    @property
    def candidate_count(self) -> int:
        """Number of candidates in this cell"""
        return len(self.candidates)
    
    def remove_candidate(self, digit: int) -> bool:
        """Remove a candidate from this cell, return True if removed"""
        if digit in self.candidates:
            self.candidates.remove(digit)
            return True
        return False
    
    def set_value(self, value: int):
        """Set the cell value and clear candidates"""
        self.value = value
        self.candidates.clear()


class Grid:
    """9x9 Sudoku grid representation"""
    
    def __init__(self, initial_grid: Optional[np.ndarray] = None):
        self.size = 9
        self.box_size = 3
        
        if initial_grid is not None:
            self.grid = initial_grid.copy()
        else:
            self.grid = np.full((self.size, self.size), None, dtype=object)
        
        # Initialize cells
        self.cells = {}
        for row in range(self.size):
            for col in range(self.size):
                value = self.grid[row, col] if self.grid[row, col] is not None else None
                self.cells[(row, col)] = Cell(row, col, value)
    
    def get_cell(self, row: int, col: int) -> Cell:
        """Get cell at given position"""
        return self.cells[(row, col)]
    
    def set_value(self, row: int, col: int, value: int):
        """Set value at given position"""
        self.grid[row, col] = value
        self.cells[(row, col)].set_value(value)
    
    def get_value(self, row: int, col: int) -> Optional[int]:
        """Get value at given position"""
        return self.grid[row, col]
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is valid"""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_box(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all positions in the same box as (row, col)"""
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        positions = []
        for r in range(box_row, box_row + self.box_size):
            for c in range(box_col, box_col + self.box_size):
                positions.append((r, c))
        return positions
    
    def get_row(self, row: int) -> List[Tuple[int, int]]:
        """Get all positions in the given row"""
        return [(row, col) for col in range(self.size)]
    
    def get_col(self, col: int) -> List[Tuple[int, int]]:
        """Get all positions in the given column"""
        return [(row, col) for row in range(self.size)]
    
    def get_peers(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all peer positions (same row, col, or box)"""
        peers = set()
        
        # Same row
        for c in range(self.size):
            if c != col:
                peers.add((row, c))
        
        # Same column
        for r in range(self.size):
            if r != row:
                peers.add((r, col))
        
        # Same box
        box_positions = self.get_box(row, col)
        for r, c in box_positions:
            if (r, c) != (row, col):
                peers.add((r, c))
        
        return peers
    
    def is_valid_move(self, row: int, col: int, value: int) -> bool:
        """Check if placing value at (row, col) would be valid"""
        if not self.is_valid_position(row, col):
            return False
        
        # Check if cell is empty
        if self.get_value(row, col) is not None:
            return False
        
        # Check row
        for c in range(self.size):
            if c != col and self.get_value(row, c) == value:
                return False
        
        # Check column
        for r in range(self.size):
            if r != row and self.get_value(r, col) == value:
                return False
        
        # Check box
        box_positions = self.get_box(row, col)
        for r, c in box_positions:
            if (r, c) != (row, col) and self.get_value(r, c) == value:
                return False
        
        return True
    
    def is_solved(self) -> bool:
        """Check if the grid is completely solved"""
        for row in range(self.size):
            for col in range(self.size):
                if self.get_value(row, col) is None:
                    return False
        return True
    
    def is_valid(self) -> bool:
        """Check if the current grid state is valid (no conflicts)"""
        # Check rows
        for row in range(self.size):
            values = [self.get_value(row, col) for col in range(self.size)]
            values = [v for v in values if v is not None]
            if len(values) != len(set(values)):
                return False
        
        # Check columns
        for col in range(self.size):
            values = [self.get_value(row, col) for row in range(self.size)]
            values = [v for v in values if v is not None]
            if len(values) != len(set(values)):
                return False
        
        # Check boxes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                values = []
                for r in range(box_row, box_row + self.box_size):
                    for c in range(box_col, box_col + self.box_size):
                        value = self.get_value(r, c)
                        if value is not None:
                            values.append(value)
                if len(values) != len(set(values)):
                    return False
        
        return True
    
    def copy(self) -> 'Grid':
        """Create a deep copy of the grid"""
        new_grid = Grid()
        new_grid.grid = self.grid.copy()
        new_grid.cells = {}
        for (row, col), cell in self.cells.items():
            new_cell = Cell(row, col, cell.value, cell.candidates.copy())
            new_grid.cells[(row, col)] = new_cell
        return new_grid
    
    def to_string(self) -> str:
        """Convert grid to string representation"""
        result = []
        for row in range(self.size):
            row_str = []
            for col in range(self.size):
                value = self.get_value(row, col)
                row_str.append(str(value) if value is not None else '.')
            result.append(''.join(row_str))
        return ''.join(result)
    
    def to_display_string(self) -> str:
        """Convert grid to formatted display string"""
        result = []
        for row in range(self.size):
            if row > 0 and row % 3 == 0:
                result.append('-' * 21)
            
            row_str = []
            for col in range(self.size):
                if col > 0 and col % 3 == 0:
                    row_str.append('|')
                
                value = self.get_value(row, col)
                row_str.append(str(value) if value is not None else '.')
            
            result.append(' '.join(row_str))
        
        return '\n'.join(result)
    
    @classmethod
    def from_string(cls, grid_str: str) -> 'Grid':
        """Create grid from string representation"""
        grid_str = grid_str.replace('\n', '').replace(' ', '').replace('.', '0')
        
        if len(grid_str) != 81:
            raise ValueError(f"Grid string must be exactly 81 characters, got {len(grid_str)}")
        
        grid = np.full((9, 9), None, dtype=object)
        for i, char in enumerate(grid_str):
            row, col = i // 9, i % 9
            value = int(char) if char != '0' else None
            grid[row, col] = value
        
        return cls(grid)
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get all empty cell positions"""
        empty = []
        for row in range(self.size):
            for col in range(self.size):
                if self.get_value(row, col) is None:
                    empty.append((row, col))
        return empty
    
    def get_filled_cells(self) -> List[Tuple[int, int]]:
        """Get all filled cell positions"""
        filled = []
        for row in range(self.size):
            for col in range(self.size):
                if self.get_value(row, col) is not None:
                    filled.append((row, col))
        return filled

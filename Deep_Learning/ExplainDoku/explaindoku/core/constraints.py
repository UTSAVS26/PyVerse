"""
Constraint management for Sudoku solving
"""

from typing import List, Tuple, Set, Dict, Optional
from .grid import Grid, Cell


class ConstraintManager:
    """Manages Sudoku constraints and candidate sets"""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.size = grid.size
        self.box_size = grid.box_size
        
        # Pre-compute all units and peers
        self._compute_units()
        self._compute_peers()
        
        # Initialize candidate sets
        self._initialize_candidates()
    
    def _compute_units(self):
        """Pre-compute all units (rows, columns, boxes)"""
        self.units = {
            'row': [],
            'col': [],
            'box': []
        }
        
        # Rows
        for row in range(self.size):
            self.units['row'].append([(row, col) for col in range(self.size)])
        
        # Columns
        for col in range(self.size):
            self.units['col'].append([(row, col) for row in range(self.size)])
        
        # Boxes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box_positions = []
                for r in range(box_row, box_row + self.box_size):
                    for c in range(box_col, box_col + self.box_size):
                        box_positions.append((r, c))
                self.units['box'].append(box_positions)
    
    def _compute_peers(self):
        """Pre-compute all peer relationships"""
        self.peers = {}
        for row in range(self.size):
            for col in range(self.size):
                self.peers[(row, col)] = self.grid.get_peers(row, col)
    
    def _initialize_candidates(self):
        """Initialize candidate sets for all cells"""
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid.get_cell(row, col)
                if not cell.is_filled:
                    # Start with all candidates
                    cell.candidates = set(range(1, 10))
                    
                    # Remove candidates that conflict with filled cells
                    for peer_row, peer_col in self.peers[(row, col)]:
                        peer_value = self.grid.get_value(peer_row, peer_col)
                        if peer_value is not None:
                            cell.remove_candidate(peer_value)
    
    def get_units_for_cell(self, row: int, col: int) -> Dict[str, List[Tuple[int, int]]]:
        """Get all units containing the given cell"""
        units = {}
        
        # Row unit
        units['row'] = self.units['row'][row]
        
        # Column unit
        units['col'] = self.units['col'][col]
        
        # Box unit
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        box_index = (box_row // self.box_size) * 3 + (box_col // self.box_size)
        units['box'] = self.units['box'][box_index]
        
        return units
    
    def get_peers(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all peer positions for the given cell"""
        return self.peers[(row, col)]
    
    def get_candidates(self, row: int, col: int) -> Set[int]:
        """Get current candidates for the given cell"""
        return self.grid.get_cell(row, col).candidates.copy()
    
    def remove_candidate(self, row: int, col: int, digit: int) -> bool:
        """Remove a candidate from a cell, return True if removed"""
        return self.grid.get_cell(row, col).remove_candidate(digit)
    
    def set_value(self, row: int, col: int, value: int):
        """Set a value and update all affected candidates"""
        # Set the value
        self.grid.set_value(row, col, value)
        
        # Remove this value from all peer candidates
        for peer_row, peer_col in self.peers[(row, col)]:
            self.remove_candidate(peer_row, peer_col, value)
    
    def get_cells_with_candidate(self, digit: int, unit_type: str, unit_index: int) -> List[Tuple[int, int]]:
        """Get all cells in a unit that have the given digit as a candidate"""
        cells = []
        unit_positions = self.units[unit_type][unit_index]
        
        for row, col in unit_positions:
            if digit in self.get_candidates(row, col):
                cells.append((row, col))
        
        return cells
    
    def get_cells_without_candidate(self, digit: int, unit_type: str, unit_index: int) -> List[Tuple[int, int]]:
        """Get all cells in a unit that don't have the given digit as a candidate"""
        cells = []
        unit_positions = self.units[unit_type][unit_index]
        
        for row, col in unit_positions:
            if digit not in self.get_candidates(row, col):
                cells.append((row, col))
        
        return cells
    
    def get_digit_frequency_in_unit(self, digit: int, unit_type: str, unit_index: int) -> int:
        """Get how many cells in a unit have the given digit as a candidate"""
        return len(self.get_cells_with_candidate(digit, unit_type, unit_index))
    
    def get_unit_entropy(self, unit_type: str, unit_index: int) -> float:
        """Calculate entropy of candidate distribution in a unit"""
        import math
        
        unit_positions = self.units[unit_type][unit_index]
        total_candidates = 0
        filled_cells = 0
        
        for row, col in unit_positions:
            cell = self.grid.get_cell(row, col)
            if cell.is_filled:
                filled_cells += 1
            else:
                total_candidates += len(cell.candidates)
        
        empty_cells = len(unit_positions) - filled_cells
        if empty_cells == 0:
            return 0.0
        
        avg_candidates = total_candidates / empty_cells
        return -math.log(avg_candidates) if avg_candidates > 0 else 0.0
    
    def get_mrv_cells(self) -> List[Tuple[int, int]]:
        """Get cells ordered by Minimum Remaining Values (MRV)"""
        empty_cells = self.grid.get_empty_cells()
        return sorted(empty_cells, key=lambda pos: len(self.get_candidates(*pos)))
    
    def get_degree_heuristic_cells(self) -> List[Tuple[int, int]]:
        """Get cells ordered by degree heuristic (number of unassigned peers)"""
        empty_cells = self.grid.get_empty_cells()
        
        def degree_heuristic(pos):
            row, col = pos
            unassigned_peers = 0
            for peer_row, peer_col in self.peers[(row, col)]:
                if not self.grid.get_cell(peer_row, peer_col).is_filled:
                    unassigned_peers += 1
            return unassigned_peers
        
        return sorted(empty_cells, key=degree_heuristic, reverse=True)
    
    def get_least_constraining_values(self, row: int, col: int) -> List[int]:
        """Get values ordered by least constraining value heuristic"""
        candidates = self.get_candidates(row, col)
        
        def lcv_heuristic(digit):
            constraints = 0
            for peer_row, peer_col in self.peers[(row, col)]:
                if digit in self.get_candidates(peer_row, peer_col):
                    constraints += 1
            return constraints
        
        return sorted(candidates, key=lcv_heuristic)
    
    def is_arc_consistent(self) -> bool:
        """Check if the grid is arc consistent (AC-3)"""
        # This is a simplified check - full AC-3 would be more complex
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid.get_cell(row, col)
                if not cell.is_filled and len(cell.candidates) == 0:
                    return False
        return True
    
    def enforce_arc_consistency(self) -> bool:
        """Enforce arc consistency using AC-3 algorithm"""
        # Simplified AC-3 implementation
        queue = []
        
        # Initialize queue with all arcs
        for row in range(self.size):
            for col in range(self.size):
                for peer_row, peer_col in self.peers[(row, col)]:
                    queue.append(((row, col), (peer_row, peer_col)))
        
        while queue:
            (row1, col1), (row2, col2) = queue.pop(0)
            
            if self._revise(row1, col1, row2, col2):
                if len(self.get_candidates(row1, col1)) == 0:
                    return False  # Inconsistent
                
                # Add arcs back to queue
                for peer_row, peer_col in self.peers[(row1, col1)]:
                    if (peer_row, peer_col) != (row2, col2):
                        queue.append(((peer_row, peer_col), (row1, col1)))
        
        return True
    
    def _revise(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        """Revise Dom(X1) wrt Dom(X2) for the constraint X1 != X2."""
        cell1 = self.grid.get_cell(row1, col1)
        cell2 = self.grid.get_cell(row2, col2)

        revised = False

        # If X2 is filled with v, remove v from Dom(X1)
        if cell2.is_filled:
            v = cell2.value
            if v in cell1.candidates and not cell1.is_filled:
                cell1.remove_candidate(v)
                return True

        if cell1.is_filled or cell2.is_filled:
            return revised

        # Otherwise, if Dom(X2) == {d}, remove d from Dom(X1)
        if len(cell2.candidates) == 1:
            (d,) = tuple(cell2.candidates)
            if d in cell1.candidates:
                cell1.remove_candidate(d)
                revised = True

        return revised
    def get_conflicts(self) -> List[Tuple[int, int]]:
        """Get all cells that have conflicts (no candidates but not filled)"""
        conflicts = []
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid.get_cell(row, col)
                if not cell.is_filled and len(cell.candidates) == 0:
                    conflicts.append((row, col))
        return conflicts
    
    def get_singleton_cells(self) -> List[Tuple[int, int]]:
        """Get all cells that have exactly one candidate (naked singles)"""
        singletons = []
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid.get_cell(row, col)
                if not cell.is_filled and len(cell.candidates) == 1:
                    singletons.append((row, col))
        return singletons
    
    def get_hidden_singles(self) -> List[Tuple[str, int, int, Tuple[int, int]]]:
        """Get all hidden singles (digit, unit_type, unit_index, cell_position)"""
        hidden_singles = []
        
        for unit_type in ['row', 'col', 'box']:
            for unit_index, unit_positions in enumerate(self.units[unit_type]):
                for digit in range(1, 10):
                    cells_with_digit = self.get_cells_with_candidate(digit, unit_type, unit_index)
                    if len(cells_with_digit) == 1:
                        cell_pos = cells_with_digit[0]
                        cell = self.grid.get_cell(*cell_pos)
                        if not cell.is_filled:
                            hidden_singles.append((digit, unit_type, unit_index, cell_pos))
        
        return hidden_singles

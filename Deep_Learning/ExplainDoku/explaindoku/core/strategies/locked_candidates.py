"""
Locked Candidates strategies (Pointing and Claiming)
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ..constraints import ConstraintManager


@dataclass
class LockedCandidateResult:
    """Result of applying a locked candidates strategy"""
    technique: str
    digit: int
    unit_type: str
    unit_index: int
    eliminations: List[Tuple[Tuple[int, int], int]]
    locked_cells: List[Tuple[int, int]]
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []
        if self.locked_cells is None:
            self.locked_cells = []


class LockedCandidatesStrategy:
    """Implements Locked Candidates strategies (Pointing and Claiming)"""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.cm = constraint_manager
    
    def find_pointing_candidates(self) -> List[LockedCandidateResult]:
        """Find pointing candidates (digit locked to one row/col within a box)"""
        results = []
        
        # Check each box
        for box_index, box_positions in enumerate(self.cm.units['box']):
            for digit in range(1, 10):
                # Find cells in this box that have this digit as candidate
                cells_with_digit = []
                for row, col in box_positions:
                    if digit in self.cm.get_candidates(row, col):
                        cells_with_digit.append((row, col))
                
                if len(cells_with_digit) >= 2:
                    # Check if all cells are in the same row
                    rows = set(row for row, col in cells_with_digit)
                    if len(rows) == 1:
                        row = rows.pop()
                        eliminations = self._get_pointing_eliminations(row, digit, 'row', cells_with_digit)
                        if eliminations:
                            results.append(LockedCandidateResult(
                                technique="pointing_row",
                                digit=digit,
                                unit_type="row",
                                unit_index=row,
                                eliminations=eliminations,
                                locked_cells=cells_with_digit
                            ))
                    
                    # Check if all cells are in the same column
                    cols = set(col for row, col in cells_with_digit)
                    if len(cols) == 1:
                        col = cols.pop()
                        eliminations = self._get_pointing_eliminations(col, digit, 'col', cells_with_digit)
                        if eliminations:
                            results.append(LockedCandidateResult(
                                technique="pointing_col",
                                digit=digit,
                                unit_type="col",
                                unit_index=col,
                                eliminations=eliminations,
                                locked_cells=cells_with_digit
                            ))
        
        return results
    
    def find_claiming_candidates(self) -> List[LockedCandidateResult]:
        """Find claiming candidates (digit locked to one box within a row/col)"""
        results = []
        
        # Check each row
        for row_index, row_positions in enumerate(self.cm.units['row']):
            for digit in range(1, 10):
                # Find cells in this row that have this digit as candidate
                cells_with_digit = []
                for row, col in row_positions:
                    if digit in self.cm.get_candidates(row, col):
                        cells_with_digit.append((row, col))
                
                if len(cells_with_digit) >= 2:
                    # Check if all cells are in the same box
                    boxes = set()
                    for r, c in cells_with_digit:
                        box_row = (r // 3) * 3
                        box_col = (c // 3) * 3
                        boxes.add((box_row, box_col))
                    
                    if len(boxes) == 1:
                        box_row, box_col = boxes.pop()
                        box_index = (box_row // 3) * 3 + (box_col // 3)
                        eliminations = self._get_claiming_eliminations(box_index, digit, cells_with_digit)
                        if eliminations:
                            results.append(LockedCandidateResult(
                                technique="claiming_row",
                                digit=digit,
                                unit_type="row",
                                unit_index=row_index,
                                eliminations=eliminations,
                                locked_cells=cells_with_digit
                            ))
        
        # Check each column
        for col_index, col_positions in enumerate(self.cm.units['col']):
            for digit in range(1, 10):
                # Find cells in this column that have this digit as candidate
                cells_with_digit = []
                for row, col in col_positions:
                    if digit in self.cm.get_candidates(row, col):
                        cells_with_digit.append((row, col))
                
                if len(cells_with_digit) >= 2:
                    # Check if all cells are in the same box
                    boxes = set()
                    for r, c in cells_with_digit:
                        box_row = (r // 3) * 3
                        box_col = (c // 3) * 3
                        boxes.add((box_row, box_col))
                    
                    if len(boxes) == 1:
                        box_row, box_col = boxes.pop()
                        box_index = (box_row // 3) * 3 + (box_col // 3)
                        eliminations = self._get_claiming_eliminations(box_index, digit, cells_with_digit)
                        if eliminations:
                            results.append(LockedCandidateResult(
                                technique="claiming_col",
                                digit=digit,
                                unit_type="col",
                                unit_index=col_index,
                                eliminations=eliminations,
                                locked_cells=cells_with_digit
                            ))
        
        return results
    
    def _get_pointing_eliminations(self, unit_index: int, digit: int, unit_type: str, locked_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], int]]:
        """Get eliminations for pointing candidates"""
        eliminations = []
        unit_positions = self.cm.units[unit_type][unit_index]
        
        for row, col in unit_positions:
            if (row, col) not in locked_cells and digit in self.cm.get_candidates(row, col):
                eliminations.append(((row, col), digit))
        
        return eliminations
    
    def _get_claiming_eliminations(self, box_index: int, digit: int, locked_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], int]]:
        """Get eliminations for claiming candidates"""
        eliminations = []
        box_positions = self.cm.units['box'][box_index]
        
        for row, col in box_positions:
            if (row, col) not in locked_cells and digit in self.cm.get_candidates(row, col):
                eliminations.append(((row, col), digit))
        
        return eliminations
    
    def apply_locked_candidates(self, result: LockedCandidateResult) -> bool:
        """Apply a locked candidates elimination"""
        if not result.eliminations:
            return False
        
        # Apply all eliminations
        for (row, col), digit in result.eliminations:
            self.cm.remove_candidate(row, col, digit)
        
        return True
    
    def find_all_locked_candidates(self) -> List[LockedCandidateResult]:
        """Find all locked candidates (pointing and claiming)"""
        results = []
        
        # Find pointing candidates
        pointing_results = self.find_pointing_candidates()
        results.extend(pointing_results)
        
        # Find claiming candidates
        claiming_results = self.find_claiming_candidates()
        results.extend(claiming_results)
        
        return results
    
    def apply_locked_candidates_strategy(self) -> List[LockedCandidateResult]:
        """Apply all possible locked candidates and return results"""
        applied_results = []
        
        while True:
            # Find all locked candidates
            locked_candidates = self.find_all_locked_candidates()
            
            if not locked_candidates:
                break
            
            # Apply the first one found
            result = locked_candidates[0]
            if self.apply_locked_candidates(result):
                applied_results.append(result)
            else:
                break
        
        return applied_results
    
    def get_locked_candidates_explanation(self, result: LockedCandidateResult) -> str:
        """Generate human-readable explanation for a locked candidates result"""
        digit = result.digit
        unit_name = self._get_unit_name(result.unit_type, result.unit_index)
        locked_positions = [f"R{r+1}C{c+1}" for r, c in result.locked_cells]
        elimination_positions = [f"R{r+1}C{c+1}" for (r, c), d in result.eliminations]
        
        if result.technique.startswith("pointing"):
            box_name = self._get_box_name(result.locked_cells[0])
            return f"Locked Candidates (Pointing): In {box_name}, digit {digit} appears only in {unit_name} at {', '.join(locked_positions)} → eliminate {digit} from {', '.join(elimination_positions)}."
        
        elif result.technique.startswith("claiming"):
            box_name = self._get_box_name(result.locked_cells[0])
            return f"Locked Candidates (Claiming): In {unit_name}, digit {digit} appears only in {box_name} at {', '.join(locked_positions)} → eliminate {digit} from {', '.join(elimination_positions)}."
        
        return f"Locked Candidates: Eliminate {digit} from {', '.join(elimination_positions)}."
    
    def _get_unit_name(self, unit_type: str, unit_index: int) -> str:
        """Get human-readable unit name"""
        if unit_type == "row":
            return f"Row {unit_index + 1}"
        elif unit_type == "col":
            return f"Column {unit_index + 1}"
        else:
            return f"{unit_type.title()} {unit_index + 1}"
    
    def _get_box_name(self, cell_position: Tuple[int, int]) -> str:
        """Get human-readable box name"""
        row, col = cell_position
        box_row = (row // 3) * 3 + 1
        box_col = (col // 3) * 3 + 1
        return f"Box ({box_row}-{box_row+2}, {box_col}-{box_col+2})"

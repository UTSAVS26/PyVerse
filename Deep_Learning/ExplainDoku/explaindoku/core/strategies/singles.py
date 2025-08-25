"""
Naked and Hidden Singles strategies
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ..constraints import ConstraintManager


@dataclass
class SingleResult:
    """Result of applying a singles strategy"""
    technique: str
    cell_position: Tuple[int, int]
    value: int
    unit_type: Optional[str] = None
    unit_index: Optional[int] = None
    eliminations: List[Tuple[Tuple[int, int], int]] = None
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []


class SinglesStrategy:
    """Implements Naked and Hidden Singles strategies"""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.cm = constraint_manager
    
    def find_naked_singles(self) -> List[SingleResult]:
        """Find all naked singles (cells with exactly one candidate)"""
        results = []
        singleton_cells = self.cm.get_singleton_cells()
        
        for row, col in singleton_cells:
            cell = self.cm.grid.get_cell(row, col)
            value = next(iter(cell.candidates))  # Get the single candidate
            
            result = SingleResult(
                technique="naked_single",
                cell_position=(row, col),
                value=value
            )
            results.append(result)
        
        return results
    
    def find_hidden_singles(self) -> List[SingleResult]:
        """Find all hidden singles (digits that can only go in one cell in a unit)"""
        results = []
        hidden_singles = self.cm.get_hidden_singles()
        
        for digit, unit_type, unit_index, cell_pos in hidden_singles:
            result = SingleResult(
                technique="hidden_single",
                cell_position=cell_pos,
                value=digit,
                unit_type=unit_type,
                unit_index=unit_index
            )
            results.append(result)
        
        return results
    
    def apply_naked_single(self, result: SingleResult) -> bool:
        """Apply a naked single placement"""
        row, col = result.cell_position
        value = result.value
        
        # Check if the placement is still valid
        if not self.cm.grid.is_valid_move(row, col, value):
            return False
        
        # Set the value
        self.cm.set_value(row, col, value)
        return True
    
    def apply_hidden_single(self, result: SingleResult) -> bool:
        """Apply a hidden single placement"""
        row, col = result.cell_position
        value = result.value
        
        # Check if the placement is still valid
        if not self.cm.grid.is_valid_move(row, col, value):
            return False
        
        # Set the value
        self.cm.set_value(row, col, value)
        return True
    
    def find_all_singles(self) -> List[SingleResult]:
        """Find all singles (naked and hidden)"""
        results = []
        
        # Find naked singles first
        naked_singles = self.find_naked_singles()
        results.extend(naked_singles)
        
        # Find hidden singles
        hidden_singles = self.find_hidden_singles()
        results.extend(hidden_singles)
        
        return results
    
    def apply_singles(self) -> List[SingleResult]:
        """Apply all possible singles and return results"""
        applied_results = []
        
        while True:
            # Find all singles
            singles = self.find_all_singles()
            
            if not singles:
                break
            
            # Apply the first single found
            single = singles[0]
            
            if single.technique == "naked_single":
                if self.apply_naked_single(single):
                    applied_results.append(single)
                else:
                    break  # Invalid placement, stop
            elif single.technique == "hidden_single":
                if self.apply_hidden_single(single):
                    applied_results.append(single)
                else:
                    break  # Invalid placement, stop
        
        return applied_results
    
    def get_singles_explanation(self, result: SingleResult) -> str:
        """Generate human-readable explanation for a singles result"""
        row, col = result.cell_position
        cell_pos = f"R{row+1}C{col+1}"
        
        if result.technique == "naked_single":
            return f"Naked Single: {cell_pos} has only candidate {result.value} → Place {result.value} in {cell_pos}."
        
        elif result.technique == "hidden_single":
            unit_name = self._get_unit_name(result.unit_type, result.unit_index)
            return f"Hidden Single: In {unit_name}, only {cell_pos} can be {result.value} → Place {result.value} in {cell_pos}."
        
        return f"Single: Place {result.value} in {cell_pos}."
    
    def _get_unit_name(self, unit_type: str, unit_index: int) -> str:
        """Get human-readable unit name"""
        if unit_type == "row":
            return f"Row {unit_index + 1}"
        elif unit_type == "col":
            return f"Column {unit_index + 1}"
        elif unit_type == "box":
            box_row = (unit_index // 3) * 3 + 1
            box_col = (unit_index % 3) * 3 + 1
            return f"Box ({box_row}-{box_row+2}, {box_col}-{box_col+2})"
        else:
            return f"{unit_type.title()} {unit_index + 1}"

"""
Fish strategies (X-Wing, Swordfish, etc.)
"""

from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from itertools import combinations
from ..constraints import ConstraintManager


@dataclass
class FishResult:
    """Result of applying a fish strategy"""
    technique: str
    digit: int
    base_units: List[int]  # Rows or columns that form the fish
    cover_units: List[int]  # Columns or rows that are covered
    eliminations: List[Tuple[Tuple[int, int], int]]
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []


class FishStrategy:
    """Implements Fish strategies (X-Wing, Swordfish, etc.)"""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.cm = constraint_manager
    
    def find_x_wing(self) -> List[FishResult]:
        """Find X-Wing patterns"""
        return self._find_fish_pattern(2, "x_wing")
    
    def find_swordfish(self) -> List[FishResult]:
        """Find Swordfish patterns"""
        return self._find_fish_pattern(3, "swordfish")
    
    def find_jellyfish(self) -> List[FishResult]:
        """Find Jellyfish patterns"""
        return self._find_fish_pattern(4, "jellyfish")
    
    def _find_fish_pattern(self, size: int, technique: str) -> List[FishResult]:
        """Find fish patterns of given size"""
        results = []
        
        # Check for fish in rows (base) covering columns (cover)
        row_fish = self._find_fish_in_units(size, 'row', 'col', technique)
        results.extend(row_fish)
        
        # Check for fish in columns (base) covering rows (cover)
        col_fish = self._find_fish_in_units(size, 'col', 'row', technique)
        results.extend(col_fish)
        
        return results
    
    def _find_fish_in_units(self, size: int, base_unit_type: str, cover_unit_type: str, technique: str) -> List[FishResult]:
        """Find fish patterns in specific unit types"""
        results = []
        
        # Get all combinations of base units
        for base_units in combinations(range(9), size):
            # For each digit, check if it forms a fish pattern
            for digit in range(1, 10):
                # Collect candidate positions in each base unit
                base_unit_covers: List[Set[int]] = []
                valid_base = True
                for u in base_units:
                    cells = self.cm.get_cells_with_candidate(digit, base_unit_type, u)
                    # cardinality constraints per base unit
                    cnt = len(cells)
                    if size == 2:
                        # X-Wing requires exactly 2 candidates per base unit
                        if cnt != 2:
                            valid_base = False
                            break
                    else:
                        # For larger fish, 2..size candidates per base unit
                        if cnt < 2 or cnt > size:
                            valid_base = False
                            break
                    # map to cover indices (columns if base units are cols, else rows)
                    covers = {c if base_unit_type == 'col' else r for r, c in cells}
                    base_unit_covers.append(covers)
                if not valid_base:
                    continue

                # Union of all cover indices must be exactly `size`
                cover_units = set().union(*base_unit_covers)
                if len(cover_units) != size:
                    continue
                # Ensure each base unit's candidates lie within those cover units
                if any(not covers.issubset(cover_units) for covers in base_unit_covers):
                    continue

                # Find eliminations based on these strictly validated fish units
                eliminations = self._get_fish_eliminations(
                    digit, cover_units, cover_unit_type, base_units
                )
                if eliminations:
                    results.append(FishResult(
                        technique=technique,
                        digit=digit,
                        base_units=list(base_units),
                        cover_units=list(cover_units),
                        eliminations=eliminations
                    ))
        
        return results
    
    def _find_cover_units_for_digit(self, digit: int, base_units: Tuple[int, ...], base_unit_type: str) -> Set[int]:
        """Find which cover units contain the given digit in the specified base units"""
        cover_units = set()
        
        for base_unit in base_units:
            # Get all cells in this base unit that have the digit as candidate
            cells_with_digit = self.cm.get_cells_with_candidate(digit, base_unit_type, base_unit)
            
            # Add the cover units for these cells
            for row, col in cells_with_digit:
                if base_unit_type == 'row':
                    cover_units.add(col)
                else:  # base_unit_type == 'col'
                    cover_units.add(row)
        
        return cover_units
    
    def _get_fish_eliminations(self, digit: int, cover_units: Set[int], cover_unit_type: str, base_units: Tuple[int, ...]) -> List[Tuple[Tuple[int, int], int]]:
        """Get eliminations for a fish pattern"""
        eliminations = []
        
        # Check all cells in the cover units that are not in the base units
        for cover_unit in cover_units:
            unit_positions = self.cm.units[cover_unit_type][cover_unit]
            
            for row, col in unit_positions:
                # Skip cells that are in the base units
                if cover_unit_type == 'col':
                    if row in base_units:
                        continue
                else:  # cover_unit_type == 'row'
                    if col in base_units:
                        continue
                
                # Check if this cell has the digit as candidate
                if digit in self.cm.get_candidates(row, col):
                    eliminations.append(((row, col), digit))
        
        return eliminations
    
    def apply_fish(self, result: FishResult) -> bool:
        """Apply a fish elimination"""
        if not result.eliminations:
            return False
        
        # Apply all eliminations
        for (row, col), digit in result.eliminations:
            self.cm.remove_candidate(row, col, digit)
        
        return True
    
    def find_all_fish(self) -> List[FishResult]:
        """Find all fish patterns"""
        results = []
        
        # Find X-Wing patterns
        x_wings = self.find_x_wing()
        results.extend(x_wings)
        
        # Find Swordfish patterns
        swordfish = self.find_swordfish()
        results.extend(swordfish)
        
        # Find Jellyfish patterns
        jellyfish = self.find_jellyfish()
        results.extend(jellyfish)
        
        return results
    
    def apply_fish_strategy(self) -> List[FishResult]:
        """Apply all possible fish patterns and return results"""
        applied_results = []
        
        while True:
            # Find all fish patterns
            fish_patterns = self.find_all_fish()
            
            if not fish_patterns:
                break
            
            # Apply the first one found
            result = fish_patterns[0]
            if self.apply_fish(result):
                applied_results.append(result)
            else:
                break
        
        return applied_results
    
    def get_fish_explanation(self, result: FishResult) -> str:
        """Generate human-readable explanation for a fish result"""
        digit = result.digit
        base_unit_name = self._get_unit_type_name(len(result.base_units))
        cover_unit_name = self._get_unit_type_name(len(result.cover_units))
        
        base_list = [str(u + 1) for u in result.base_units]
        cover_list = [str(u + 1) for u in result.cover_units]
        elimination_positions = [f"R{r+1}C{c+1}" for (r, c), d in result.eliminations]
        
        return f"{result.technique.title()}: Digit {digit} forms a {base_unit_name} in {base_unit_name} {', '.join(base_list)} covering {cover_unit_name} {', '.join(cover_list)} → eliminate {digit} from {', '.join(elimination_positions)}."
    
    def _get_unit_type_name(self, count: int) -> str:
        """Get unit type name based on count"""
        if count == 2:
            return "pair"
        elif count == 3:
            return "triple"
        elif count == 4:
            return "quadruple"
        else:
            return f"set of {count}"

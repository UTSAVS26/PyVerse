"""
Naked and Hidden Pairs/Triples strategies
"""

from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from itertools import combinations
from ..constraints import ConstraintManager


@dataclass
class PairTripleResult:
    """Result of applying a pairs/triples strategy"""
    technique: str
    digits: Set[int]
    unit_type: str
    unit_index: int
    cells: List[Tuple[int, int]]
    eliminations: List[Tuple[Tuple[int, int], int]]
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []


class PairsTriplesStrategy:
    """Implements Naked and Hidden Pairs/Triples strategies"""
    
    def __init__(self, constraint_manager: ConstraintManager):
        self.cm = constraint_manager
    
    def find_naked_pairs(self) -> List[PairTripleResult]:
        """Find naked pairs (two cells with same two candidates)"""
        return self._find_naked_subsets(2)
    
    def find_naked_triples(self) -> List[PairTripleResult]:
        """Find naked triples (three cells with same three candidates)"""
        return self._find_naked_subsets(3)
    
    def find_hidden_pairs(self) -> List[PairTripleResult]:
        """Find hidden pairs (two digits that only appear in two cells)"""
        return self._find_hidden_subsets(2)
    
    def find_hidden_triples(self) -> List[PairTripleResult]:
        """Find hidden triples (three digits that only appear in three cells)"""
        return self._find_hidden_subsets(3)
    
    def _find_naked_subsets(self, size: int) -> List[PairTripleResult]:
        """Find naked subsets of given size"""
        results = []
        
        for unit_type in ['row', 'col', 'box']:
            for unit_index, unit_positions in enumerate(self.cm.units[unit_type]):
                # Get empty cells in this unit
                empty_cells = []
                for row, col in unit_positions:
                    cell = self.cm.grid.get_cell(row, col)
                    if not cell.is_filled:
                        empty_cells.append((row, col))
                
                # Check all combinations of cells
                for cell_combo in combinations(empty_cells, size):
                    # Get candidates for these cells
                    cell_candidates = []
                    for row, col in cell_combo:
                        candidates = self.cm.get_candidates(row, col)
                        cell_candidates.append(candidates)
                    
                    # Check if all cells have the same candidates
                    if len(set(map(frozenset, cell_candidates))) == 1:
                        shared_candidates = cell_candidates[0]
                        
                        # Check if the number of candidates equals the number of cells
                        if len(shared_candidates) == size:
                            # Find eliminations
                            eliminations = self._get_naked_subset_eliminations(
                                unit_type, unit_index, cell_combo, shared_candidates
                            )
                            
                            if eliminations:
                                results.append(PairTripleResult(
                                    technique=f"naked_{self._get_size_name(size)}",
                                    digits=shared_candidates,
                                    unit_type=unit_type,
                                    unit_index=unit_index,
                                    cells=list(cell_combo),
                                    eliminations=eliminations
                                ))
        
        return results
    
    def _find_hidden_subsets(self, size: int) -> List[PairTripleResult]:
        """Find hidden subsets of given size"""
        results = []
        
        for unit_type in ['row', 'col', 'box']:
            for unit_index, unit_positions in enumerate(self.cm.units[unit_type]):
                # Get empty cells in this unit
                empty_cells = []
                for row, col in unit_positions:
                    cell = self.cm.grid.get_cell(row, col)
                    if not cell.is_filled:
                        empty_cells.append((row, col))
                
                # Check all combinations of digits
                for digit_combo in combinations(range(1, 10), size):
                    # Find cells that have any of these digits as candidates
                    cells_with_digits = set()
                    for row, col in empty_cells:
                        candidates = self.cm.get_candidates(row, col)
                        if any(digit in candidates for digit in digit_combo):
                            cells_with_digits.add((row, col))
                    
                    # Check if exactly 'size' cells have these digits
                    if len(cells_with_digits) == size:
                        # Check if these cells only have these digits as candidates
                        all_candidates = set()
                        for row, col in cells_with_digits:
                            candidates = self.cm.get_candidates(row, col)
                            all_candidates.update(candidates)
                        
                        if all_candidates.issubset(set(digit_combo)):
                            # Find eliminations
                            eliminations = self._get_hidden_subset_eliminations(
                                unit_type, unit_index, cells_with_digits, set(digit_combo)
                            )
                            
                            if eliminations:
                                results.append(PairTripleResult(
                                    technique=f"hidden_{self._get_size_name(size)}",
                                    digits=set(digit_combo),
                                    unit_type=unit_type,
                                    unit_index=unit_index,
                                    cells=list(cells_with_digits),
                                    eliminations=eliminations
                                ))
        
        return results
    
    def _get_naked_subset_eliminations(self, unit_type: str, unit_index: int, 
                                     subset_cells: Tuple, shared_candidates: Set[int]) -> List[Tuple[Tuple[int, int], int]]:
        """Get eliminations for naked subset"""
        eliminations = []
        unit_positions = self.cm.units[unit_type][unit_index]
        
        for row, col in unit_positions:
            if (row, col) not in subset_cells:
                cell = self.cm.grid.get_cell(row, col)
                if not cell.is_filled:
                    for digit in shared_candidates:
                        if digit in cell.candidates:
                            eliminations.append(((row, col), digit))
        
        return eliminations
    
    def _get_hidden_subset_eliminations(self, unit_type: str, unit_index: int,
                                      subset_cells: Set[Tuple[int, int]], subset_digits: Set[int]) -> List[Tuple[Tuple[int, int], int]]:
        """Get eliminations for hidden subset"""
        eliminations = []
        
        for row, col in subset_cells:
            cell = self.cm.grid.get_cell(row, col)
            if not cell.is_filled:
                for digit in cell.candidates:
                    if digit not in subset_digits:
                        eliminations.append(((row, col), digit))
        
        return eliminations
    
    def _get_size_name(self, size: int) -> str:
        """Get name for subset size"""
        if size == 2:
            return "pair"
        elif size == 3:
            return "triple"
        elif size == 4:
            return "quad"
        else:
            return f"subset_{size}"
    
    def apply_pairs_triples(self, result: PairTripleResult) -> bool:
        """Apply a pairs/triples elimination"""
        if not result.eliminations:
            return False
        
        # Apply all eliminations
        for (row, col), digit in result.eliminations:
            self.cm.remove_candidate(row, col, digit)
        
        return True
    
    def find_all_pairs_triples(self) -> List[PairTripleResult]:
        """Find all pairs and triples (naked and hidden)"""
        results = []
        
        # Find naked pairs and triples
        naked_pairs = self.find_naked_pairs()
        results.extend(naked_pairs)
        
        naked_triples = self.find_naked_triples()
        results.extend(naked_triples)
        
        # Find hidden pairs and triples
        hidden_pairs = self.find_hidden_pairs()
        results.extend(hidden_pairs)
        
        hidden_triples = self.find_hidden_triples()
        results.extend(hidden_triples)
        
        return results
    
    def apply_pairs_triples_strategy(self) -> List[PairTripleResult]:
        """Apply all possible pairs/triples and return results"""
        applied_results = []
        
        while True:
            # Find all pairs and triples
            pairs_triples = self.find_all_pairs_triples()
            
            if not pairs_triples:
                break
            
            # Apply the first one found
            result = pairs_triples[0]
            if self.apply_pairs_triples(result):
                applied_results.append(result)
            else:
                break
        
        return applied_results
    
    def get_pairs_triples_explanation(self, result: PairTripleResult) -> str:
        """Generate human-readable explanation for a pairs/triples result"""
        unit_name = self._get_unit_name(result.unit_type, result.unit_index)
        cell_positions = [f"R{r+1}C{c+1}" for r, c in result.cells]
        digit_list = sorted(result.digits)
        elimination_positions = [f"R{r+1}C{c+1}" for (r, c), d in result.eliminations]
        
        if result.technique.startswith("naked"):
            size_name = result.technique.split('_')[1]
            return f"Naked {size_name.title()}: In {unit_name}, cells {', '.join(cell_positions)} contain only candidates {digit_list} → eliminate {digit_list} from {', '.join(elimination_positions)}."
        
        elif result.technique.startswith("hidden"):
            size_name = result.technique.split('_')[1]
            return f"Hidden {size_name.title()}: In {unit_name}, digits {digit_list} appear only in cells {', '.join(cell_positions)} → eliminate other candidates from these cells."
        
        return f"Pairs/Triples: Eliminate candidates from {', '.join(elimination_positions)}."
    
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

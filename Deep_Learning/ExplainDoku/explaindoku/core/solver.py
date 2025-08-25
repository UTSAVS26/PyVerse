"""
Main solver that orchestrates all solving strategies
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .grid import Grid
from .constraints import ConstraintManager
from .strategies.singles import SinglesStrategy, SingleResult
from .strategies.locked_candidates import LockedCandidatesStrategy, LockedCandidateResult
from .strategies.pairs_triples import PairsTriplesStrategy, PairTripleResult
from .strategies.fish import FishStrategy, FishResult
from .search import SearchStrategy, SearchResult


@dataclass
class SolveStep:
    """A single step in the solving process"""
    step_number: int
    technique: str
    explanation: str
    cell_position: Optional[Tuple[int, int]] = None
    value: Optional[int] = None
    eliminations: List[Tuple[Tuple[int, int], int]] = None
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []


@dataclass
class SolveResult:
    """Result of solving a Sudoku puzzle"""
    success: bool
    steps: List[SolveStep]
    final_grid: Optional[Grid] = None
    total_steps: int = 0
    human_steps: int = 0
    search_steps: int = 0
    backtrack_count: int = 0


class Solver:
    """Main solver that orchestrates all solving strategies"""
    
    def __init__(self, grid: Grid):
        self.original_grid = grid
        self.cm = ConstraintManager(grid)
        self.singles_strategy = SinglesStrategy(self.cm)
        self.locked_candidates_strategy = LockedCandidatesStrategy(self.cm)
        self.pairs_triples_strategy = PairsTriplesStrategy(self.cm)
        self.fish_strategy = FishStrategy(self.cm)
        self.search_strategy = SearchStrategy(self.cm)
    
    def solve(self, use_search: bool = True, max_backtracks: int = 10000) -> SolveResult:
        """Solve the puzzle using all available strategies"""
        steps = []
        step_number = 1
        
        # Reset to original state
        self.cm = ConstraintManager(self.original_grid.copy())
        
        # Update strategy objects to use new constraint manager
        self.singles_strategy = SinglesStrategy(self.cm)
        self.locked_candidates_strategy = LockedCandidatesStrategy(self.cm)
        self.pairs_triples_strategy = PairsTriplesStrategy(self.cm)
        self.fish_strategy = FishStrategy(self.cm)
        self.search_strategy = SearchStrategy(self.cm)
        
        # Apply human strategies first
        human_steps = self._apply_human_strategies()
        steps.extend(human_steps)
        step_number += len(human_steps)
        
        # Check if solved after human strategies
        if self.cm.grid.is_solved():
            return SolveResult(
                success=True,
                steps=steps,
                final_grid=self.cm.grid,
                total_steps=len(steps),
                human_steps=len(human_steps),
                search_steps=0,
                backtrack_count=0
            )
        
        # Use search if requested and puzzle not solved
        if use_search and not self.cm.grid.is_solved():
            search_result = self._apply_search_strategy(max_backtracks)
            if search_result.success:
                steps.extend(search_result.steps)
                return SolveResult(
                    success=True,
                    steps=steps,
                    final_grid=search_result.final_grid,
                    total_steps=len(steps),
                    human_steps=len(human_steps),
                    search_steps=len(search_result.steps),
                    backtrack_count=search_result.backtrack_count
                )
        
        # If we get here, puzzle couldn't be solved
        return SolveResult(
            success=False,
            steps=steps,
            final_grid=self.cm.grid,
            total_steps=len(steps),
            human_steps=len(human_steps),
            search_steps=0,
            backtrack_count=0
        )
    
    def solve_step_by_step(self) -> List[SolveStep]:
        """Solve step by step, returning each step as it's applied"""
        steps = []
        step_number = 1
        
        # Reset to original state
        self.cm = ConstraintManager(self.original_grid.copy())
        
        # Update strategy objects to use new constraint manager
        self.singles_strategy = SinglesStrategy(self.cm)
        self.locked_candidates_strategy = LockedCandidatesStrategy(self.cm)
        self.pairs_triples_strategy = PairsTriplesStrategy(self.cm)
        self.fish_strategy = FishStrategy(self.cm)
        self.search_strategy = SearchStrategy(self.cm)
        
        while not self.cm.grid.is_solved():
            # Try human strategies first
            step = self._get_next_human_step()
            
            if step is None:
                # No human strategies available, try search
                step = self._get_next_search_step()
                
                if step is None:
                    # No more steps possible
                    break
            
            step.step_number = step_number
            steps.append(step)
            step_number += 1
        
        return steps
    
    def _apply_human_strategies(self) -> List[SolveStep]:
        """Apply all human strategies and return steps"""
        steps = []
        step_number = 1
        
        while True:
            # Try singles first
            singles_results = self.singles_strategy.apply_singles()
            for result in singles_results:
                steps.append(SolveStep(
                    step_number=step_number,
                    technique=result.technique,
                    explanation=self.singles_strategy.get_singles_explanation(result),
                    cell_position=result.cell_position,
                    value=result.value,
                    eliminations=result.eliminations
                ))
                step_number += 1
            
            # Try locked candidates
            locked_results = self.locked_candidates_strategy.apply_locked_candidates_strategy()
            for result in locked_results:
                steps.append(SolveStep(
                    step_number=step_number,
                    technique=result.technique,
                    explanation=self.locked_candidates_strategy.get_locked_candidates_explanation(result),
                    eliminations=result.eliminations
                ))
                step_number += 1
            
            # Try pairs and triples
            pairs_results = self.pairs_triples_strategy.apply_pairs_triples_strategy()
            for result in pairs_results:
                steps.append(SolveStep(
                    step_number=step_number,
                    technique=result.technique,
                    explanation=self.pairs_triples_strategy.get_pairs_triples_explanation(result),
                    eliminations=result.eliminations
                ))
                step_number += 1
            
            # Try fish strategies
            fish_results = self.fish_strategy.apply_fish_strategy()
            for result in fish_results:
                steps.append(SolveStep(
                    step_number=step_number,
                    technique=result.technique,
                    explanation=self.fish_strategy.get_fish_explanation(result),
                    eliminations=result.eliminations
                ))
                step_number += 1
            
            # Check if any progress was made
            if not (singles_results or locked_results or pairs_results or fish_results):
                break
            
            # Check if solved after each round
            if self.cm.grid.is_solved():
                break
        
        return steps
    
    def _apply_search_strategy(self, max_backtracks: int) -> SolveResult:
        """Apply search strategy and return result"""
        # Create a copy of current state for search
        search_grid = self.cm.grid.copy()
        search_cm = ConstraintManager(search_grid)
        search_strategy = SearchStrategy(search_cm)
        
        # Try to solve with backtracking
        solved_grid = search_strategy.solve_with_backtracking(max_backtracks)
        
        if solved_grid is not None:
            return SolveResult(
                success=True,
                steps=[],  # Search steps are not individually tracked
                final_grid=solved_grid,
                total_steps=0,
                human_steps=0,
                search_steps=1,  # Count as one search step
                backtrack_count=search_strategy.backtrack_count
            )
        else:
            return SolveResult(
                success=False,
                steps=[],
                final_grid=search_grid,
                total_steps=0,
                human_steps=0,
                search_steps=0,
                backtrack_count=search_strategy.backtrack_count
            )
    
    def _get_next_human_step(self) -> Optional[SolveStep]:
        """Get the next human strategy step"""
        # Try singles
        singles = self.singles_strategy.find_all_singles()
        if singles:
            result = singles[0]
            if result.technique == "naked_single":
                self.singles_strategy.apply_naked_single(result)
            elif result.technique == "hidden_single":
                self.singles_strategy.apply_hidden_single(result)
            
            return SolveStep(
                step_number=0,  # Will be set by caller
                technique=result.technique,
                explanation=self.singles_strategy.get_singles_explanation(result),
                cell_position=result.cell_position,
                value=result.value,
                eliminations=result.eliminations
            )
        
        # Try locked candidates
        locked = self.locked_candidates_strategy.find_all_locked_candidates()
        if locked:
            result = locked[0]
            self.locked_candidates_strategy.apply_locked_candidates(result)
            
            return SolveStep(
                step_number=0,  # Will be set by caller
                technique=result.technique,
                explanation=self.locked_candidates_strategy.get_locked_candidates_explanation(result),
                eliminations=result.eliminations
            )
        
        # Try pairs and triples
        pairs_triples = self.pairs_triples_strategy.find_all_pairs_triples()
        if pairs_triples:
            result = pairs_triples[0]
            self.pairs_triples_strategy.apply_pairs_triples(result)
            
            return SolveStep(
                step_number=0,  # Will be set by caller
                technique=result.technique,
                explanation=self.pairs_triples_strategy.get_pairs_triples_explanation(result),
                eliminations=result.eliminations
            )
        
        # Try fish strategies
        fish = self.fish_strategy.find_all_fish()
        if fish:
            result = fish[0]
            self.fish_strategy.apply_fish(result)
            
            return SolveStep(
                step_number=0,  # Will be set by caller
                technique=result.technique,
                explanation=self.fish_strategy.get_fish_explanation(result),
                eliminations=result.eliminations
            )
        
        return None
    
    def _get_next_search_step(self) -> Optional[SolveStep]:
        """Get the next search step"""
        # This is a simplified version - in practice, you'd want to track search steps more carefully
        empty_cells = self.cm.grid.get_empty_cells()
        if not empty_cells:
            return None
        
        # Get next cell using MRV
        next_cell = min(empty_cells, key=lambda pos: len(self.cm.get_candidates(*pos)))
        row, col = next_cell
        
        candidates = self.cm.get_candidates(row, col)
        if not candidates:
            return None
        
        # Try the first candidate
        value = min(candidates)
        self.cm.set_value(row, col, value)
        
        return SolveStep(
            step_number=0,  # Will be set by caller
            technique="backtracking",
            explanation=f"Backtracking: Try {value} in R{row+1}C{col+1}",
            cell_position=(row, col),
            value=value
        )
    
    def get_difficulty_estimate(self) -> str:
        """Estimate difficulty from the strongest technique used (no search)."""
        result = Solver(self.original_grid).solve(use_search=False)
        if not result.success:
            return "Hard"
        techniques = {step.technique for step in result.steps}
        # If any fish is used → Hard
        if any(t in {"x_wing", "swordfish", "jellyfish"} for t in techniques):
            return "Hard"
        # If any subsets/locked-candidates are used → Medium
        if any(t in {
            "pointing_row", "pointing_col", "claiming_row", "claiming_col",
            "naked_pair", "hidden_pair", "naked_triple", "hidden_triple"
        } for t in techniques):
            return "Medium"
        # Otherwise singles only → Easy
        return "Easy"

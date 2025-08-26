"""
Tests for the Solver module
"""

import pytest
from explaindoku.core.grid import Grid
from explaindoku.core.solver import Solver, SolveStep, SolveResult


class TestSolver:
    """Test Solver class"""
    
    def test_solver_creation(self):
        """Test solver creation"""
        grid = Grid()
        solver = Solver(grid)
        
        assert solver.original_grid is not None
        assert solver.cm is not None
        assert solver.singles_strategy is not None
        assert solver.locked_candidates_strategy is not None
        assert solver.pairs_triples_strategy is not None
        assert solver.fish_strategy is not None
        assert solver.search_strategy is not None
    
    def test_solve_easy_puzzle(self):
        """Test solving an easy puzzle"""
        # Create a simple puzzle that can be solved with singles
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        result = solver.solve(use_search=False)
        
        assert result.success
        assert result.total_steps > 0
        assert result.human_steps > 0
        assert result.search_steps == 0
        assert result.final_grid is not None
        assert result.final_grid.is_solved()
    
    def test_solve_with_search(self):
        """Test solving with search enabled"""
        # Use a puzzle that can be solved with search
        # This is a known solvable puzzle that requires backtracking
        grid_str = "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        result = solver.solve(use_search=True)
        
        # Should be able to solve it
        assert result.success
        assert result.final_grid.is_solved()
        assert result.final_grid.is_valid()
        # The puzzle should be solved, whether by human strategies or search
        assert result.total_steps > 0
    
    def test_solve_without_search(self):
        """Test solving without search"""
        # Create a puzzle that requires search
        grid_str = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        result = solver.solve(use_search=False)
        
        # May not be able to solve without search
        # Just check that the solver doesn't crash
        assert isinstance(result, SolveResult)
        assert result.final_grid is not None
    
    def test_solve_step_by_step(self):
        """Test step-by-step solving"""
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        steps = solver.solve_step_by_step()
        
        assert isinstance(steps, list)
        assert len(steps) > 0
        
        for step in steps:
            assert isinstance(step, SolveStep)
            assert step.technique is not None
            assert step.explanation is not None
    
    def test_get_next_human_step(self):
        """Test getting next human step"""
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        step = solver._get_next_human_step()
        
        if step is not None:
            assert isinstance(step, SolveStep)
            assert step.technique is not None
            assert step.explanation is not None
    
    def test_get_next_search_step(self):
        """Test getting next search step"""
        grid_str = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        step = solver._get_next_search_step()
        
        if step is not None:
            assert isinstance(step, SolveStep)
            assert step.technique == "backtracking"
    
    def test_difficulty_estimate(self):
        """Test difficulty estimation"""
        # Easy puzzle
        easy_grid = Grid.from_string("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
        easy_solver = Solver(easy_grid)
        easy_difficulty = easy_solver.get_difficulty_estimate()
        
        # Hard puzzle
        hard_grid = Grid.from_string("800000000003600000070090200050007000000045700000100030001000068008500010090000400")
        hard_solver = Solver(hard_grid)
        hard_difficulty = hard_solver.get_difficulty_estimate()
        
        assert easy_difficulty in ["Easy", "Medium", "Hard"]
        assert hard_difficulty in ["Easy", "Medium", "Hard"]
    
    def test_unsolvable_puzzle(self):
        """Test handling of unsolvable puzzle"""
        # Create an invalid puzzle (duplicate in first row)
        grid_str = "550000000000000000000000000000000000000000000000000000000000000000000000000000000"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        result = solver.solve(use_search=True)
        
        # Should not be able to solve
        assert not result.success
    
    def test_already_solved_puzzle(self):
        """Test solving an already solved puzzle"""
        solved_str = "123456789456789123789123456234567891567891234891234567345678912678912345912345678"
        grid = Grid.from_string(solved_str)
        solver = Solver(grid)
        
        result = solver.solve()
        
        assert result.success
        assert result.total_steps == 0
        assert result.final_grid.is_solved()
    
    def test_empty_puzzle(self):
        """Test solving an empty puzzle"""
        grid = Grid()
        solver = Solver(grid)
        
        result = solver.solve(use_search=True)
        
        # Should be able to solve empty puzzle with search
        assert result.success
        assert result.final_grid.is_solved()
    
    def test_max_backtracks_limit(self):
        """Test max backtracks limit"""
        # Create a very hard puzzle
        grid_str = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
        grid = Grid.from_string(grid_str)
        solver = Solver(grid)
        
        # Try with very low backtrack limit
        result = solver.solve(use_search=True, max_backtracks=1)
        
        # Should not be able to solve with such low limit
        assert not result.success


class TestSolveStep:
    """Test SolveStep class"""
    
    def test_solve_step_creation(self):
        """Test SolveStep creation"""
        step = SolveStep(
            step_number=1,
            technique="naked_single",
            explanation="Test explanation",
            cell_position=(0, 0),
            value=5
        )
        
        assert step.step_number == 1
        assert step.technique == "naked_single"
        assert step.explanation == "Test explanation"
        assert step.cell_position == (0, 0)
        assert step.value == 5
        assert step.eliminations == []
    
    def test_solve_step_with_eliminations(self):
        """Test SolveStep with eliminations"""
        eliminations = [((1, 1), 5), ((2, 2), 5)]
        step = SolveStep(
            step_number=2,
            technique="locked_candidates",
            explanation="Test explanation",
            eliminations=eliminations
        )
        
        assert step.eliminations == eliminations


class TestSolveResult:
    """Test SolveResult class"""
    
    def test_solve_result_creation(self):
        """Test SolveResult creation"""
        grid = Grid()
        steps = []
        
        result = SolveResult(
            success=True,
            steps=steps,
            final_grid=grid,
            total_steps=0,
            human_steps=0,
            search_steps=0,
            backtrack_count=0
        )
        
        assert result.success
        assert result.steps == steps
        assert result.final_grid == grid
        assert result.total_steps == 0
        assert result.human_steps == 0
        assert result.search_steps == 0
        assert result.backtrack_count == 0

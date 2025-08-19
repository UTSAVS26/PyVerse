"""
Tests for the solver module.
"""

import pytest
from unittest.mock import Mock, patch
from riddlemind.solver import RiddleMind, ReasoningEngine, Solution, ReasoningStep
from riddlemind.constraints import (
    ConstraintSet, ConstraintType, ConstraintBuilder
)


class TestRiddleMind:
    """Test the RiddleMind class."""
    
    def test_riddlemind_initialization(self):
        """Test RiddleMind initialization."""
        solver = RiddleMind()
        assert solver is not None
        assert hasattr(solver, 'parser')
        assert hasattr(solver, 'reasoning_engine')
    
    def test_solve_simple_puzzle(self):
        """Test solving a simple puzzle."""
        solver = RiddleMind()
        
        puzzle_text = "Alice is older than Bob. Charlie is younger than Alice. Who is the oldest?"
        solution = solver.solve(puzzle_text)
        
        assert isinstance(solution, Solution)
        assert solution.original_text == puzzle_text
        assert len(solution.parsed_constraints) == 2
        assert len(solution.questions) == 1
        assert "Who is the oldest?" in solution.questions
    
    def test_solve_complex_puzzle(self):
        """Test solving a complex puzzle."""
        solver = RiddleMind()
        
        puzzle_text = """
        Alice is older than Bob. 
        Bob is taller than Charlie. 
        Alice is Bob's sister. 
        Who is the oldest? 
        Who is the tallest?
        """
        
        solution = solver.solve(puzzle_text)
        
        assert isinstance(solution, Solution)
        assert len(solution.parsed_constraints) >= 3
        assert len(solution.questions) == 2
        
        # Check that conclusions are generated
        assert len(solution.conclusions) > 0
    
    def test_solve_empty_puzzle(self):
        """Test solving an empty puzzle."""
        solver = RiddleMind()
        
        solution = solver.solve("")
        
        assert isinstance(solution, Solution)
        assert len(solution.parsed_constraints) == 0
        assert len(solution.questions) == 0
    
    def test_solve_puzzle_with_contradictions(self):
        """Test solving a puzzle with contradictions."""
        solver = RiddleMind()
        
        puzzle_text = "Alice is older than Bob. Bob is older than Alice."
        solution = solver.solve(puzzle_text)
        
        assert isinstance(solution, Solution)
        assert len(solution.errors) > 0  # Should detect contradictions
    
    def test_solve_interactive_mode(self):
        """Test interactive mode (mocked)."""
        solver = RiddleMind()
        
        # Mock input to simulate user interaction
        with patch('builtins.input', side_effect=['Alice is older than Bob', '']):
            solution = solver.solve_interactive()
            
            assert isinstance(solution, Solution)
            assert "Alice is older than Bob" in solution.original_text


class TestReasoningEngine:
    """Test the ReasoningEngine class."""
    
    def test_reasoning_engine_initialization(self):
        """Test ReasoningEngine initialization."""
        engine = ReasoningEngine()
        assert engine is not None
        assert engine.step_counter == 0
    
    def test_next_step(self):
        """Test step counter increment."""
        engine = ReasoningEngine()
        
        assert engine._next_step() == 1
        assert engine._next_step() == 2
        assert engine._next_step() == 3
    
    def test_reason_with_transitivity(self):
        """Test reasoning with transitivity."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # A > B, B > C should imply A > C
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Bob", "Charlie")
        )
        
        reasoning_steps, conclusions = engine.reason(constraint_set)
        
        assert len(reasoning_steps) > 0
        assert len(conclusions) > 0
        
        # Should find transitivity
        transitivity_found = any("Derived: older(Alice, Charlie)" in conclusion 
                               for conclusion in conclusions)
        assert transitivity_found
    
    def test_reason_with_contradictions(self):
        """Test reasoning with contradictions."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Add contradictory constraints
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("younger", "Bob", "Alice")
        )
        
        reasoning_steps, conclusions = engine.reason(constraint_set)
        
        assert len(reasoning_steps) > 0
        assert len(conclusions) > 0
        
        # Should detect contradictions
        contradiction_found = any("CONTRADICTION" in conclusion 
                                for conclusion in conclusions)
        assert contradiction_found
    
    def test_find_extrema(self):
        """Test finding extreme values."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Alice > Bob, Bob > Charlie
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Bob", "Charlie")
        )
        
        extrema_conclusions = engine._find_extrema(constraint_set)
        
        assert len(extrema_conclusions) > 0
        assert any("Alice is the oldest" in conclusion for conclusion in extrema_conclusions)
    
    def test_find_extrema_no_clear_winner(self):
        """Test finding extrema when there's no clear winner."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Alice > Bob, Charlie > Bob (no comparison between Alice and Charlie)
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Charlie", "Bob")
        )
        
        extrema_conclusions = engine._find_extrema(constraint_set)
        
        # Should find candidates but not a clear winner
        assert len(extrema_conclusions) > 0
        assert any("candidates" in conclusion.lower() for conclusion in extrema_conclusions)
    
    def test_analyze_relationships(self):
        """Test analyzing relationships."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Add family relationships
        constraint_set.add_constraint(
            ConstraintBuilder.create_relationship_constraint("brother", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_relationship_constraint("sister", "Bob", "Charlie")
        )
        
        relationship_conclusions = engine._analyze_relationships(constraint_set)
        
        assert len(relationship_conclusions) > 0
        assert any("Family members" in conclusion for conclusion in relationship_conclusions)
    
    def test_analyze_spatial_relationships(self):
        """Test analyzing spatial relationships."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Add spatial relationships
        constraint_set.add_constraint(
            ConstraintBuilder.create_relationship_constraint("left_of", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_relationship_constraint("right_of", "Charlie", "Bob")
        )
        
        relationship_conclusions = engine._analyze_relationships(constraint_set)
        
        assert len(relationship_conclusions) > 0
        assert any("Spatial arrangement" in conclusion for conclusion in relationship_conclusions)
    
    def test_generate_summary(self):
        """Test generating summary conclusions."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        # Add various types of constraints
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_relationship_constraint("brother", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_equality_constraint("Charlie", "David")
        )
        
        summary_conclusions = engine._generate_summary(constraint_set)
        
        assert len(summary_conclusions) > 0
        assert any("comparison constraints" in conclusion for conclusion in summary_conclusions)
        assert any("relationship constraints" in conclusion for conclusion in summary_conclusions)
        assert any("equality constraints" in conclusion for conclusion in summary_conclusions)
        assert any("Entities involved" in conclusion for conclusion in summary_conclusions)
    
    def test_reason_empty_constraint_set(self):
        """Test reasoning with empty constraint set."""
        engine = ReasoningEngine()
        
        constraint_set = ConstraintSet()
        
        reasoning_steps, conclusions = engine.reason(constraint_set)
        
        assert len(reasoning_steps) == 0
        assert len(conclusions) == 0


class TestSolution:
    """Test the Solution class."""
    
    def test_solution_creation(self):
        """Test Solution object creation."""
        constraint_set = ConstraintSet()
        reasoning_steps = []
        conclusions = []
        
        solution = Solution(
            original_text="Test puzzle",
            parsed_constraints=constraint_set,
            questions=["Who is the oldest?"],
            reasoning_steps=reasoning_steps,
            conclusions=conclusions,
            warnings=[],
            errors=[]
        )
        
        assert solution.original_text == "Test puzzle"
        assert len(solution.parsed_constraints) == 0
        assert len(solution.questions) == 1
        assert len(solution.reasoning_steps) == 0
        assert len(solution.conclusions) == 0
        assert len(solution.warnings) == 0
        assert len(solution.errors) == 0
    
    def test_solution_string_representation(self):
        """Test Solution string representation."""
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                description="Test step",
                input_constraints=[],
                derived_constraints=[],
                reasoning_type="test"
            )
        ]
        
        solution = Solution(
            original_text="Alice is older than Bob. Who is the oldest?",
            parsed_constraints=constraint_set,
            questions=["Who is the oldest?"],
            reasoning_steps=reasoning_steps,
            conclusions=["Alice is the oldest"],
            warnings=[],
            errors=[]
        )
        
        solution_str = str(solution)
        
        assert "RiddleMind Solution" in solution_str
        assert "Alice is older than Bob" in solution_str
        assert "older(Alice, Bob)" in solution_str
        assert "Who is the oldest?" in solution_str
        assert "Test step" in solution_str
        assert "Alice is the oldest" in solution_str


class TestReasoningStep:
    """Test the ReasoningStep class."""
    
    def test_reasoning_step_creation(self):
        """Test ReasoningStep object creation."""
        step = ReasoningStep(
            step_number=1,
            description="Test reasoning step",
            input_constraints=[],
            derived_constraints=[],
            reasoning_type="transitivity"
        )
        
        assert step.step_number == 1
        assert step.description == "Test reasoning step"
        assert len(step.input_constraints) == 0
        assert len(step.derived_constraints) == 0
        assert step.reasoning_type == "transitivity"
    
    def test_reasoning_step_with_constraints(self):
        """Test ReasoningStep with constraints."""
        input_constraint = ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        derived_constraint = ConstraintBuilder.create_comparison_constraint("older", "Alice", "Charlie")
        
        step = ReasoningStep(
            step_number=1,
            description="Applying transitivity",
            input_constraints=[input_constraint],
            derived_constraints=[derived_constraint],
            reasoning_type="transitivity"
        )
        
        assert len(step.input_constraints) == 1
        assert len(step.derived_constraints) == 1
        assert step.input_constraints[0].predicate == "older"
        assert step.derived_constraints[0].predicate == "older"


class TestIntegration:
    """Integration tests for the complete solving process."""
    
    def test_complete_solving_process(self):
        """Test the complete solving process from text to solution."""
        solver = RiddleMind()
        
        puzzle_text = """
        Alice is older than Bob.
        Bob is taller than Charlie.
        Alice is Bob's sister.
        Who is the oldest?
        Who is the tallest?
        """
        
        solution = solver.solve(puzzle_text)
        
        # Check that parsing worked
        assert len(solution.parsed_constraints) >= 3
        
        # Check that questions were identified
        assert len(solution.questions) == 2
        
        # Check that reasoning was applied
        assert len(solution.reasoning_steps) > 0
        
        # Check that conclusions were drawn
        assert len(solution.conclusions) > 0
        
        # Check that no errors occurred
        assert len(solution.errors) == 0
    
    def test_solving_with_validation_issues(self):
        """Test solving with validation issues."""
        solver = RiddleMind()
        
        # This should trigger warnings about few entities
        puzzle_text = "Alice is older than Bob."
        
        solution = solver.solve(puzzle_text)
        
        # Should have warnings but no errors
        assert len(solution.warnings) > 0
        assert len(solution.errors) == 0
    
    def test_solving_complex_transitive_relationships(self):
        """Test solving with complex transitive relationships."""
        solver = RiddleMind()
        
        puzzle_text = """
        Alice is older than Bob.
        Bob is older than Charlie.
        Charlie is older than David.
        Who is the oldest?
        Who is the youngest?
        """
        
        solution = solver.solve(puzzle_text)
        
        # Should find transitivity
        assert len(solution.reasoning_steps) > 0
        
        # Should identify oldest and youngest
        conclusions_text = " ".join(solution.conclusions)
        assert "Alice is the oldest" in conclusions_text or "oldest" in conclusions_text.lower()
        assert "David is the youngest" in conclusions_text or "youngest" in conclusions_text.lower()

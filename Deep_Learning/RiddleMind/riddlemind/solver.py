"""
Logic reasoning engine for RiddleMind.

This module implements the core reasoning engine that processes logical constraints
and derives conclusions using symbolic reasoning.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from .constraints import (
    Constraint, ConstraintSet, ConstraintType, 
    ConstraintBuilder, ConstraintValidator
)
from .parser import PuzzleParser


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    
    step_number: int
    description: str
    input_constraints: List[Constraint]
    derived_constraints: List[Constraint]
    reasoning_type: str  # "transitivity", "contradiction", "inference", etc.


@dataclass
class Solution:
    """Represents a complete solution to a logic puzzle."""
    
    original_text: str
    parsed_constraints: ConstraintSet
    questions: List[str]
    reasoning_steps: List[ReasoningStep]
    conclusions: List[str]
    warnings: List[str]
    errors: List[str]
    
    def __str__(self) -> str:
        """String representation of the solution."""
        output = []
        output.append("🧠 RiddleMind Solution")
        output.append("=" * 50)
        
        output.append("\n📝 Original Puzzle:")
        output.append(self.original_text)
        
        output.append("\n🔍 Parsed Constraints:")
        if self.parsed_constraints:
            for constraint in self.parsed_constraints:
                output.append(f"  - {constraint}")
        else:
            output.append("  No constraints parsed")
        
        output.append("\n❓ Questions:")
        for question in self.questions:
            output.append(f"  - {question}")
        
        if self.reasoning_steps:
            output.append("\n🧮 Reasoning Steps:")
            for step in self.reasoning_steps:
                output.append(f"  {step.step_number}. {step.description}")
                if step.derived_constraints:
                    for constraint in step.derived_constraints:
                        output.append(f"     → {constraint}")
        
        if self.conclusions:
            output.append("\n✅ Conclusions:")
            for conclusion in self.conclusions:
                output.append(f"  - {conclusion}")
        
        if self.warnings:
            output.append("\n⚠️  Warnings:")
            for warning in self.warnings:
                output.append(f"  - {warning}")
        
        if self.errors:
            output.append("\n❌ Errors:")
            for error in self.errors:
                output.append(f"  - {error}")
        
        return "\n".join(output)


class RiddleMind:
    """Main logic puzzle solver class."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the RiddleMind solver."""
        self.parser = PuzzleParser(spacy_model)
        self.reasoning_engine = ReasoningEngine()
    
    def solve(self, puzzle_text: str) -> Solution:
        """
        Solve a logic puzzle given in natural language.
        
        Args:
            puzzle_text: The puzzle text in natural language
            
        Returns:
            Solution object containing the complete analysis
        """
        # Parse the puzzle
        parsed_constraints, questions = self.parser.parse(puzzle_text)
        
        # Validate parsing
        validation_issues = self.parser.validate_parsing(parsed_constraints)
        
        # Apply reasoning
        reasoning_steps, conclusions = self.reasoning_engine.reason(parsed_constraints)
        
        # Generate solution
        solution = Solution(
            original_text=puzzle_text,
            parsed_constraints=parsed_constraints,
            questions=questions,
            reasoning_steps=reasoning_steps,
            conclusions=conclusions,
            warnings=validation_issues.get("warnings", []),
            errors=validation_issues.get("errors", [])
        )
        
        return solution
    
    def solve_interactive(self) -> Solution:
        """Interactive mode for solving puzzles."""
        print("🧠 Welcome to RiddleMind!")
        print("Enter your logic puzzle (press Enter twice to finish):")
        
        lines = []
        while True:
            line = input()
            if line.strip() == "" and lines:
                break
            lines.append(line)
        
        puzzle_text = "\n".join(lines)
        return self.solve(puzzle_text)


class ReasoningEngine:
    """Core reasoning engine that applies logical rules."""
    
    def __init__(self):
        """Initialize the reasoning engine."""
        self.step_counter = 0
    
    def reason(self, constraint_set: ConstraintSet) -> Tuple[List[ReasoningStep], List[str]]:
        """
        Apply logical reasoning to derive conclusions.
        
        Args:
            constraint_set: Set of parsed constraints
            
        Returns:
            Tuple of (reasoning_steps, conclusions)
        """
        reasoning_steps = []
        conclusions = []
        
        # Step 1: Check for contradictions
        contradictions = ConstraintValidator.check_contradictions(constraint_set)
        if contradictions:
            step = ReasoningStep(
                step_number=self._next_step(),
                description="Checking for contradictions",
                input_constraints=[],
                derived_constraints=[],
                reasoning_type="contradiction_check"
            )
            reasoning_steps.append(step)
            
            for c1, c2 in contradictions:
                conclusions.append(f"CONTRADICTION: {c1} contradicts {c2}")
        
        # Step 2: Apply transitivity
        transitive_constraints = ConstraintValidator.check_transitivity(constraint_set)
        if transitive_constraints:
            step = ReasoningStep(
                step_number=self._next_step(),
                description="Applying transitivity rules",
                input_constraints=[],
                derived_constraints=transitive_constraints,
                reasoning_type="transitivity"
            )
            reasoning_steps.append(step)
            
            for constraint in transitive_constraints:
                conclusions.append(f"Derived: {constraint}")
        
        # Step 3: Find extrema (oldest, youngest, etc.)
        extrema_conclusions = self._find_extrema(constraint_set)
        if extrema_conclusions:
            step = ReasoningStep(
                step_number=self._next_step(),
                description="Finding extrema (oldest, youngest, etc.)",
                input_constraints=[],
                derived_constraints=[],
                reasoning_type="extrema"
            )
            reasoning_steps.append(step)
            conclusions.extend(extrema_conclusions)
        
        # Step 4: Analyze relationships
        relationship_conclusions = self._analyze_relationships(constraint_set)
        if relationship_conclusions:
            step = ReasoningStep(
                step_number=self._next_step(),
                description="Analyzing relationships",
                input_constraints=[],
                derived_constraints=[],
                reasoning_type="relationships"
            )
            reasoning_steps.append(step)
            conclusions.extend(relationship_conclusions)
        
        # Step 5: Generate summary conclusions
        summary_conclusions = self._generate_summary(constraint_set)
        if summary_conclusions:
            step = ReasoningStep(
                step_number=self._next_step(),
                description="Generating summary conclusions",
                input_constraints=[],
                derived_constraints=[],
                reasoning_type="summary"
            )
            reasoning_steps.append(step)
            conclusions.extend(summary_conclusions)
        
        return reasoning_steps, conclusions
    
    def _next_step(self) -> int:
        """Get the next step number."""
        self.step_counter += 1
        return self.step_counter
    
    def _find_extrema(self, constraint_set: ConstraintSet) -> List[str]:
        """Find extreme values (oldest, youngest, tallest, etc.)."""
        conclusions = []
        
        # Find oldest/youngest
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        younger_constraints = constraint_set.get_constraints_by_predicate("younger")
        
        if older_constraints or younger_constraints:
            # Build age hierarchy
            age_graph = {}
            for constraint in older_constraints:
                if constraint.arguments[0] not in age_graph:
                    age_graph[constraint.arguments[0]] = set()
                age_graph[constraint.arguments[0]].add(constraint.arguments[1])
            
            # Find oldest (no one is older than them)
            all_entities = constraint_set.entities
            oldest_candidates = []
            youngest_candidates = []
            
            for entity in all_entities:
                is_oldest = True
                is_youngest = True
                
                for other_entity in all_entities:
                    if other_entity != entity:
                        # Check if someone is older than this entity
                        if (other_entity in age_graph and 
                            entity in age_graph[other_entity]):
                            is_oldest = False
                        
                        # Check if this entity is older than someone else
                        if (entity in age_graph and 
                            other_entity in age_graph[entity]):
                            is_youngest = False
                
                if is_oldest:
                    oldest_candidates.append(entity)
                if is_youngest:
                    youngest_candidates.append(entity)
            
            if len(oldest_candidates) == 1:
                conclusions.append(f"{oldest_candidates[0]} is the oldest")
            elif len(oldest_candidates) > 1:
                conclusions.append(f"Oldest candidates: {', '.join(oldest_candidates)}")
            
            if len(youngest_candidates) == 1:
                conclusions.append(f"{youngest_candidates[0]} is the youngest")
            elif len(youngest_candidates) > 1:
                conclusions.append(f"Youngest candidates: {', '.join(youngest_candidates)}")
        
        # Find tallest/shortest
        taller_constraints = constraint_set.get_constraints_by_predicate("taller")
        if taller_constraints:
            # Similar logic for height
            height_graph = {}
            for constraint in taller_constraints:
                if constraint.arguments[0] not in height_graph:
                    height_graph[constraint.arguments[0]] = set()
                height_graph[constraint.arguments[0]].add(constraint.arguments[1])
            
            all_entities = constraint_set.entities
            tallest_candidates = []
            
            for entity in all_entities:
                is_tallest = True
                for other_entity in all_entities:
                    if other_entity != entity:
                        if (other_entity in height_graph and 
                            entity in height_graph[other_entity]):
                            is_tallest = False
                            break
                
                if is_tallest:
                    tallest_candidates.append(entity)
            
            if len(tallest_candidates) == 1:
                conclusions.append(f"{tallest_candidates[0]} is the tallest")
            elif len(tallest_candidates) > 1:
                conclusions.append(f"Tallest candidates: {', '.join(tallest_candidates)}")
        
        return conclusions
    
    def _analyze_relationships(self, constraint_set: ConstraintSet) -> List[str]:
        """Analyze relationship constraints."""
        conclusions = []
        
        # Analyze family relationships
        brother_constraints = constraint_set.get_constraints_by_predicate("brother")
        sister_constraints = constraint_set.get_constraints_by_predicate("sister")
        
        if brother_constraints or sister_constraints:
            family_members = set()
            for constraint in brother_constraints + sister_constraints:
                family_members.update(constraint.arguments)
            
            if len(family_members) >= 2:
                conclusions.append(f"Family members identified: {', '.join(family_members)}")
        
        # Analyze spatial relationships
        left_constraints = constraint_set.get_constraints_by_predicate("left_of")
        right_constraints = constraint_set.get_constraints_by_predicate("right_of")
        
        if left_constraints or right_constraints:
            spatial_entities = set()
            for constraint in left_constraints + right_constraints:
                spatial_entities.update(constraint.arguments)
            
            if len(spatial_entities) >= 2:
                conclusions.append(f"Spatial arrangement involves: {', '.join(spatial_entities)}")
        
        return conclusions
    
    def _generate_summary(self, constraint_set: ConstraintSet) -> List[str]:
        """Generate summary conclusions about the puzzle."""
        conclusions = []
        
        # Count different types of constraints
        comparison_count = len(constraint_set.get_constraints_by_type(ConstraintType.COMPARISON))
        relationship_count = len(constraint_set.get_constraints_by_type(ConstraintType.RELATIONSHIP))
        equality_count = len(constraint_set.get_constraints_by_type(ConstraintType.EQUALITY))
        
        if comparison_count > 0:
            conclusions.append(f"Found {comparison_count} comparison constraints")
        
        if relationship_count > 0:
            conclusions.append(f"Found {relationship_count} relationship constraints")
        
        if equality_count > 0:
            conclusions.append(f"Found {equality_count} equality constraints")
        
        # Summary of entities
        if len(constraint_set.entities) > 0:
            conclusions.append(f"Entities involved: {', '.join(sorted(constraint_set.entities))}")
        
        return conclusions

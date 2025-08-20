"""
RiddleMind - Logic Puzzle Solver Bot

A powerful logic puzzle solver that uses natural language processing 
and symbolic reasoning to solve complex riddles and logic puzzles.
"""

__version__ = "1.0.0"
__author__ = "RiddleMind Team"

from .solver import RiddleMind
from .parser import PuzzleParser
from .constraints import Constraint, ConstraintSet

__all__ = [
    "RiddleMind",
    "PuzzleParser", 
    "Constraint",
    "ConstraintSet"
]

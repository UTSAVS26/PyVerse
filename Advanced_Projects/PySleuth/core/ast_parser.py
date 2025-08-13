import ast
from typing import List, Tuple

class ASTParser:
    """
    Provides static code analysis utilities for extracting conditions, assignments, and control flow from source code.
    """
    def __init__(self, source: str):
        self.tree = ast.parse(source)

    def get_conditions(self) -> List[Tuple[int, str]]:
        """Return a list of (lineno, condition) for if/while statements."""
        conditions = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.While)):
                cond = ast.unparse(node.test) if hasattr(ast, 'unparse') else ast.dump(node.test)
                conditions.append((node.lineno, cond))
        return conditions

    def get_assignments(self) -> List[Tuple[int, str]]:
        """Return a list of (lineno, assignment) for variable assignments."""
        assignments = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                targets = [ast.unparse(t) if hasattr(ast, 'unparse') else ast.dump(t) for t in node.targets]
                value = ast.unparse(node.value) if hasattr(ast, 'unparse') else ast.dump(node.value)
                assignments.append((node.lineno, f"{' = '.join(targets)} = {value}"))
        return assignments

    def get_control_flow(self) -> List[Tuple[int, str]]:
        """Return a list of (lineno, type) for control flow statements (if, for, while, function defs)."""
        flow = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.If):
                flow.append((node.lineno, 'if'))
            elif isinstance(node, ast.For):
                flow.append((node.lineno, 'for'))
            elif isinstance(node, ast.While):
                flow.append((node.lineno, 'while'))
            elif isinstance(node, ast.FunctionDef):
                flow.append((node.lineno, f'function {node.name}'))
        return flow
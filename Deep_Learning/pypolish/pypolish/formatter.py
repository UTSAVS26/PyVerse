"""
Code Formatter for PyPolish

Handles code formatting using black and isort for consistent Python code style.
"""

import black
import isort
from typing import Optional, Dict, Any
import ast


class CodeFormatter:
    """Handles code formatting and style improvements."""
    
    def __init__(self, line_length: int = 88):
        self.line_length = line_length
        self.black_mode = black.FileMode(line_length=line_length)
    
    def format_code(self, code: str) -> str:
        """Format the given code using black and isort."""
        try:
            # First, sort imports
            code = self._sort_imports(code)
            
            # Then format with black
            code = self._format_with_black(code)
            
            return code
        except Exception as e:
            # If formatting fails, return original code
            return code
    
    def _sort_imports(self, code: str) -> str:
        """Sort imports using isort."""
        try:
            return isort.code(code, profile="black", line_length=self.line_length)
        except Exception:
            return code
    
    def _format_with_black(self, code: str) -> str:
        """Format code using black."""
        try:
            return black.format_str(code, mode=self.black_mode)
        except Exception:
            return code
    
    def add_type_hints(self, code: str) -> str:
        """Add basic type hints to function definitions."""
        try:
            tree = ast.parse(code)
            modified = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Add return type hint if missing
                    if not node.returns:
                        # Try to infer return type from function body
                        return_type = self._infer_return_type(node)
                        if return_type:
                            node.returns = ast.Name(id=return_type, ctx=ast.Load())
                            modified = True
                    
                    # Add parameter type hints if missing
                    for arg in node.args.args:
                        if not arg.annotation:
                            # Try to infer parameter type from usage
                            param_type = self._infer_parameter_type(arg.arg, node)
                            if param_type:
                                arg.annotation = ast.Name(id=param_type, ctx=ast.Load())
                                modified = True
            
            if modified:
                return ast.unparse(tree)
            return code
        except Exception:
            return code
    
    def _infer_return_type(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Infer return type from function body."""
        # Simple heuristics for return type inference
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value is None:
                    return "None"
                elif isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, bool):
                        return "bool"
                    elif isinstance(node.value.value, int):
                        return "int"
                    elif isinstance(node.value.value, float):
                        return "float"
                    elif isinstance(node.value.value, str):
                        return "str"
                elif isinstance(node.value, ast.List):
                    return "List"
                elif isinstance(node.value, ast.Dict):
                    return "Dict"
                elif isinstance(node.value, ast.Tuple):
                    return "Tuple"
        
        # Check for print statements (likely returns None)
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                return "None"
        
        return None
    
    def _infer_parameter_type(self, param_name: str, func_node: ast.FunctionDef) -> Optional[str]:
        """Infer parameter type from usage in function."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                # Likely integer operation
                return "int"
            elif isinstance(node, ast.Compare):
                # Check comparison operations
                for op in node.ops:
                    if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        return "int"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'len':
                    return "List"
                elif isinstance(node.func, ast.Name) and node.func.id == 'str':
                    return "str"
                elif isinstance(node.func, ast.Name) and node.func.id == 'int':
                    return "str"
        
        return None
    
    def add_docstrings(self, code: str) -> str:
        """Add basic docstrings to functions and classes."""
        try:
            tree = ast.parse(code)
            modified = False
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                        # Add simple docstring
                        docstring = self._generate_docstring(node)
                        if docstring:
                            docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                            node.body.insert(0, docstring_node)
                            modified = True
            
            if modified:
                return ast.unparse(tree)
            return code
        except Exception:
            return code
    
    def _generate_docstring(self, node: ast.AST) -> Optional[str]:
        """Generate a simple docstring for a function or class."""
        if isinstance(node, ast.FunctionDef):
            return f"TODO: Add docstring for {node.name} function."
        elif isinstance(node, ast.ClassDef):
            return f"TODO: Add docstring for {node.name} class."
        return None
    
    def convert_to_ternary(self, code: str) -> str:
        """Convert simple if/else statements to ternary expressions."""
        try:
            tree = ast.parse(code)
            modified = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if (len(node.body) == 1 and len(node.orelse) == 1 and
                        isinstance(node.body[0], ast.Expr) and isinstance(node.orelse[0], ast.Expr)):
                        # Convert to ternary
                        ternary = ast.IfExp(
                            test=node.test,
                            body=node.body[0].value,
                            orelse=node.orelse[0].value
                        )
                        # Replace the if statement with the ternary
                        # This is a simplified approach - in practice, you'd need more complex logic
                        modified = True
            
            if modified:
                return ast.unparse(tree)
            return code
        except Exception:
            return code
    
    def remove_unused_imports(self, code: str, unused_imports: list) -> str:
        """Remove unused imports from the code."""
        if not unused_imports:
            return code
        
        try:
            lines = code.split('\n')
            new_lines = []
            skip_lines = set()
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Check if this line contains an unused import
                should_skip = False
                for unused_import in unused_imports:
                    if unused_import in line_stripped and (line_stripped.startswith('import ') or line_stripped.startswith('from ')):
                        should_skip = True
                        break
                
                if should_skip:
                    skip_lines.add(i)
                else:
                    new_lines.append(line)
            
            return '\n'.join(new_lines)
        except Exception:
            return code

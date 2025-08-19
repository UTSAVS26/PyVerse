"""
AST Analyzer for PyPolish

Analyzes Python code using AST to identify potential improvements and anti-patterns.
"""

import ast
from typing import List, Dict, Any, Set, Tuple
import re


class ASTAnalyzer:
    """Analyzes Python code using AST to identify improvements."""
    
    def __init__(self):
        self.issues = []
        self.suggestions = []
        self.used_names = set()
        self.imported_names = set()
        self.function_definitions = []
        self.class_definitions = []
        
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze the given Python code and return analysis results."""
        try:
            tree = ast.parse(code)
            self._reset_state()
            self._analyze_node(tree)
            return {
                'issues': self.issues,
                'suggestions': self.suggestions,
                'used_names': self.used_names,
                'imported_names': self.imported_names,
                'function_count': len(self.function_definitions),
                'class_count': len(self.class_definitions),
            }
        except SyntaxError as e:
            return {
                'error': f"Syntax error: {e}",
                'issues': [],
                'suggestions': [],
            }
    
    def _reset_state(self):
        """Reset the analyzer state."""
        self.issues = []
        self.suggestions = []
        self.used_names = set()
        self.imported_names = set()
        self.function_definitions = []
        self.class_definitions = []
    
    def _analyze_node(self, node: ast.AST):
        """Recursively analyze AST nodes."""
        if isinstance(node, ast.Module):
            for child in node.body:
                self._analyze_node(child)
        elif isinstance(node, ast.Import):
            self._analyze_import(node)
        elif isinstance(node, ast.ImportFrom):
            self._analyze_import_from(node)
        elif isinstance(node, ast.FunctionDef):
            self._analyze_function_def(node)
            # Recursively analyze function body
            for child in node.body:
                self._analyze_node(child)
        elif isinstance(node, ast.ClassDef):
            self._analyze_class_def(node)
            # Recursively analyze class body
            for child in node.body:
                self._analyze_node(child)
        elif isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.For):
            self._analyze_for(node)
        elif isinstance(node, ast.While):
            self._analyze_while(node)
        elif isinstance(node, ast.Name):
            self._analyze_name(node)
        elif isinstance(node, ast.Call):
            self._analyze_call(node)
    
    def _analyze_import(self, node: ast.Import):
        """Analyze import statements."""
        for alias in node.names:
            self.imported_names.add(alias.name)
            if alias.asname:
                self.imported_names.add(alias.asname)
    
    def _analyze_import_from(self, node: ast.ImportFrom):
        """Analyze from-import statements."""
        for alias in node.names:
            # For from imports, we need to track the actual imported name
            imported_name = alias.name
            if alias.asname:
                imported_name = alias.asname
            self.imported_names.add(imported_name)
    
    def _analyze_function_def(self, node: ast.FunctionDef):
        """Analyze function definitions."""
        self.function_definitions.append(node)
        
        # Check for missing type hints
        if not node.returns and not node.name.startswith('_'):
            self.suggestions.append({
                'type': 'missing_type_hint',
                'message': f"Consider adding return type hint to function '{node.name}'",
                'line': node.lineno,
                'severity': 'info'
            })
        
        # Check for missing docstring
        if not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            self.suggestions.append({
                'type': 'missing_docstring',
                'message': f"Consider adding a docstring to function '{node.name}'",
                'line': node.lineno,
                'severity': 'info'
            })
        
        # Check for long functions
        function_lines = self._count_lines(node)
        if function_lines > 20:
            self.issues.append({
                'type': 'long_function',
                'message': f"Function '{node.name}' is {function_lines} lines long. Consider breaking it into smaller functions.",
                'line': node.lineno,
                'severity': 'warning'
            })
    
    def _analyze_class_def(self, node: ast.ClassDef):
        """Analyze class definitions."""
        self.class_definitions.append(node)
        
        # Check for missing docstring
        if not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            self.suggestions.append({
                'type': 'missing_docstring',
                'message': f"Consider adding a docstring to class '{node.name}'",
                'line': node.lineno,
                'severity': 'info'
            })
    
    def _analyze_assign(self, node: ast.Assign):
        """Analyze assignment statements."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check for unused variables
                if target.id.startswith('_'):
                    continue
                # This will be checked later when we analyze name usage
    
    def _analyze_if(self, node: ast.If):
        """Analyze if statements."""
        # Check for simple if/else that could be ternary
        if (len(node.body) == 1 and len(node.orelse) == 1 and
            isinstance(node.body[0], ast.Expr) and isinstance(node.orelse[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Return) and isinstance(node.orelse[0].value, ast.Return)):
            self.suggestions.append({
                'type': 'ternary_expression',
                'message': "Consider using ternary expression instead of if/else",
                'line': node.lineno,
                'severity': 'info'
            })
    
    def _analyze_for(self, node: ast.For):
        """Analyze for loops."""
        # Check for list comprehensions opportunities
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Call) and
            hasattr(node.body[0].value.func, 'value') and
            hasattr(node.body[0].value.func.value, 'id') and
            node.body[0].value.func.value.id in ['append', 'extend']):
            self.suggestions.append({
                'type': 'list_comprehension',
                'message': "Consider using list comprehension instead of append in loop",
                'line': node.lineno,
                'severity': 'info'
            })
    
    def _analyze_while(self, node: ast.While):
        """Analyze while loops."""
        # Check for infinite loops
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.issues.append({
                'type': 'infinite_loop',
                'message': "Potential infinite loop detected",
                'line': node.lineno,
                'severity': 'warning'
            })
    
    def _analyze_name(self, node: ast.Name):
        """Analyze name usage."""
        self.used_names.add(node.id)
    
    def _analyze_call(self, node: ast.Call):
        """Analyze function calls."""
        if isinstance(node.func, ast.Name):
            # Check for print statements (suggest logging)
            if node.func.id == 'print':
                self.suggestions.append({
                    'type': 'use_logging',
                    'message': "Consider using logging instead of print for better debugging",
                    'line': node.lineno,
                    'severity': 'info'
                })
    
    def _count_lines(self, node: ast.AST) -> int:
        """Count the number of lines in an AST node."""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        # Fallback: count the number of statements in the function
        if isinstance(node, ast.FunctionDef):
            return len(node.body)
        return 1
    
    def get_unused_imports(self) -> List[str]:
        """Get list of unused imports."""
        return list(self.imported_names - self.used_names)
    
    def get_undefined_names(self) -> List[str]:
        """Get list of undefined names (excluding builtins)."""
        builtins = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
            'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
            'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
            'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
            'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview',
            'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property',
            'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted',
            'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        return list(self.used_names - self.imported_names - builtins)

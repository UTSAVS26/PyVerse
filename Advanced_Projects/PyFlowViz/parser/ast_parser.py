"""
AST-based parser for Python code analysis.
"""

import ast
import inspect
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class FlowNode:
    """Represents a node in the flowchart."""
    id: str
    label: str
    node_type: str
    line_number: int
    children: List[str] = None
    parent: Optional[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class FlowEdge:
    """Represents an edge in the flowchart."""
    source: str
    target: str
    label: str = ""
    condition: str = ""


class ASTParser:
    """Parser that uses Python's Abstract Syntax Tree to analyze code structure."""
    
    def __init__(self):
        self.nodes: Dict[str, FlowNode] = {}
        self.edges: List[FlowEdge] = []
        self.node_counter = 0
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Python file and return flowchart data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            self._parse_ast(tree, file_path)
            
            return {
                'nodes': list(self.nodes.values()),
                'edges': self.edges,
                'source_code': source_code
            }
        except Exception as e:
            raise ValueError(f"Failed to parse file {file_path}: {str(e)}")
    
    def parse_string(self, source_code: str) -> Dict[str, Any]:
        """Parse Python code string and return flowchart data."""
        try:
            tree = ast.parse(source_code)
            self._parse_ast(tree, "string_input")
            
            return {
                'nodes': list(self.nodes.values()),
                'edges': self.edges,
                'source_code': source_code
            }
        except Exception as e:
            raise ValueError(f"Failed to parse source code: {str(e)}")
    
    def _parse_ast(self, tree: ast.AST, source_name: str):
        """Parse AST tree and build flowchart structure."""
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        
        # Add start node
        start_node = self._create_node("Start", "start", 1)
        
        # Parse all top-level statements
        for stmt in tree.body:
            self._parse_statement(stmt, start_node.id)
        
        # Add end node if there are nodes
        if self.nodes:
            end_node = self._create_node("End", "end", max(n.line_number for n in self.nodes.values()) + 1)
            # Connect last nodes to end
            for node in self.nodes.values():
                if not node.children and node.node_type != "end":
                    self._create_edge(node.id, end_node.id)
    
    def _parse_statement(self, stmt: ast.stmt, parent_id: str):
        """Parse a single AST statement."""
        if isinstance(stmt, ast.FunctionDef):
            self._parse_function_def(stmt, parent_id)
        elif isinstance(stmt, ast.If):
            self._parse_if_statement(stmt, parent_id)
        elif isinstance(stmt, ast.For):
            self._parse_for_loop(stmt, parent_id)
        elif isinstance(stmt, ast.While):
            self._parse_while_loop(stmt, parent_id)
        elif isinstance(stmt, ast.Try):
            self._parse_try_block(stmt, parent_id)
        elif isinstance(stmt, ast.Expr):
            self._parse_expression(stmt.value, parent_id)
        else:
            # Generic statement
            node = self._create_node(
                f"Statement: {type(stmt).__name__}",
                "statement",
                getattr(stmt, 'lineno', 1)
            )
            self._create_edge(parent_id, node.id)
    
    def _parse_function_def(self, func: ast.FunctionDef, parent_id: str):
        """Parse function definition."""
        func_node = self._create_node(
            f"def {func.name}()",
            "function",
            func.lineno
        )
        self._create_edge(parent_id, func_node.id)
        
        # Parse function body
        for stmt in func.body:
            self._parse_statement(stmt, func_node.id)
    
    def _parse_if_statement(self, if_stmt: ast.If, parent_id: str):
        """Parse if-elif-else statement."""
        # Create if condition node
        condition_text = self._get_condition_text(if_stmt.test)
        if_node = self._create_node(
            f"if {condition_text}",
            "condition",
            if_stmt.lineno
        )
        self._create_edge(parent_id, if_node.id, "True")
        
        # Parse if body
        for stmt in if_stmt.body:
            self._parse_statement(stmt, if_node.id)
        
        # Handle elif and else
        if if_stmt.orelse:
            if len(if_stmt.orelse) == 1 and isinstance(if_stmt.orelse[0], ast.If):
                # This is an elif
                self._parse_if_statement(if_stmt.orelse[0], parent_id)
            else:
                # This is an else
                else_node = self._create_node("else", "condition", if_stmt.orelse[0].lineno)
                self._create_edge(parent_id, else_node.id, "False")
                
                for stmt in if_stmt.orelse:
                    self._parse_statement(stmt, else_node.id)
    
    def _parse_for_loop(self, for_stmt: ast.For, parent_id: str):
        """Parse for loop."""
        target_text = self._get_target_text(for_stmt.target)
        iter_text = self._get_expression_text(for_stmt.iter)
        
        loop_node = self._create_node(
            f"for {target_text} in {iter_text}",
            "loop",
            for_stmt.lineno
        )
        self._create_edge(parent_id, loop_node.id)
        
        # Parse loop body
        for stmt in for_stmt.body:
            self._parse_statement(stmt, loop_node.id)
        
        # Connect back to loop for iteration
        self._create_edge(loop_node.id, loop_node.id, "next")
        
        # Parse else clause if exists
        if for_stmt.orelse:
            else_node = self._create_node("else", "condition", for_stmt.orelse[0].lineno)
            self._create_edge(loop_node.id, else_node.id, "done")
            
            for stmt in for_stmt.orelse:
                self._parse_statement(stmt, else_node.id)
    
    def _parse_while_loop(self, while_stmt: ast.While, parent_id: str):
        """Parse while loop."""
        condition_text = self._get_condition_text(while_stmt.test)
        
        loop_node = self._create_node(
            f"while {condition_text}",
            "loop",
            while_stmt.lineno
        )
        self._create_edge(parent_id, loop_node.id)
        
        # Parse loop body
        for stmt in while_stmt.body:
            self._parse_statement(stmt, loop_node.id)
        
        # Connect back to loop for iteration
        self._create_edge(loop_node.id, loop_node.id, "True")
        
        # Parse else clause if exists
        if while_stmt.orelse:
            else_node = self._create_node("else", "condition", while_stmt.orelse[0].lineno)
            self._create_edge(loop_node.id, else_node.id, "False")
            
            for stmt in while_stmt.orelse:
                self._parse_statement(stmt, else_node.id)
    
    def _parse_try_block(self, try_stmt: ast.Try, parent_id: str):
        """Parse try-except block."""
        try_node = self._create_node("try", "try", try_stmt.lineno)
        self._create_edge(parent_id, try_node.id)
        
        # Parse try body
        for stmt in try_stmt.body:
            self._parse_statement(stmt, try_node.id)
        
        # Parse except handlers
        for handler in try_stmt.handlers:
            if handler.type:
                except_text = f"except {self._get_expression_text(handler.type)}"
            else:
                except_text = "except"
            
            except_node = self._create_node(except_text, "except", handler.lineno)
            self._create_edge(try_node.id, except_node.id)
            
            for stmt in handler.body:
                self._parse_statement(stmt, except_node.id)
        
        # Parse finally if exists
        if try_stmt.finalbody:
            finally_node = self._create_node("finally", "finally", try_stmt.finalbody[0].lineno)
            self._create_edge(try_node.id, finally_node.id)
            
            for stmt in try_stmt.finalbody:
                self._parse_statement(stmt, finally_node.id)
    
    def _parse_expression(self, expr: ast.expr, parent_id: str):
        """Parse expression statement."""
        if isinstance(expr, ast.Call):
            func_name = self._get_expression_text(expr.func)
            node = self._create_node(
                f"Call: {func_name}()",
                "function_call",
                getattr(expr, 'lineno', 1)
            )
            self._create_edge(parent_id, node.id)
        else:
            # Generic expression
            expr_text = self._get_expression_text(expr)
            node = self._create_node(
                f"Expression: {expr_text}",
                "expression",
                getattr(expr, 'lineno', 1)
            )
            self._create_edge(parent_id, node.id)
    
    def _create_node(self, label: str, node_type: str, line_number: int) -> FlowNode:
        """Create a new flowchart node."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        node = FlowNode(
            id=node_id,
            label=label,
            node_type=node_type,
            line_number=line_number
        )
        
        self.nodes[node_id] = node
        return node
    
    def _create_edge(self, source: str, target: str, label: str = ""):
        """Create a new flowchart edge."""
        edge = FlowEdge(source=source, target=target, label=label)
        self.edges.append(edge)
        
        # Update node children/parent
        if source in self.nodes:
            self.nodes[source].children.append(target)
        if target in self.nodes:
            self.nodes[target].parent = source
    
    def _get_condition_text(self, test: ast.expr) -> str:
        """Get text representation of a condition."""
        return self._get_expression_text(test)
    
    def _get_target_text(self, target: ast.expr) -> str:
        """Get text representation of a target."""
        return self._get_expression_text(target)
    
    def _get_expression_text(self, expr: ast.expr) -> str:
        """Get text representation of an expression."""
        if isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.Constant):
            return repr(expr.value)
        elif isinstance(expr, ast.Attribute):
            return f"{self._get_expression_text(expr.value)}.{expr.attr}"
        elif isinstance(expr, ast.Call):
            func_text = self._get_expression_text(expr.func)
            args_text = ", ".join(self._get_expression_text(arg) for arg in expr.args)
            return f"{func_text}({args_text})"
        elif isinstance(expr, ast.Compare):
            left = self._get_expression_text(expr.left)
            ops = [op.__class__.__name__ for op in expr.ops]
            comparators = [self._get_expression_text(comp) for comp in expr.comparators]
            return f"{left} {' '.join(ops)} {' '.join(comparators)}"
        elif isinstance(expr, ast.BinOp):
            left = self._get_expression_text(expr.left)
            right = self._get_expression_text(expr.right)
            op = type(expr.op).__name__
            return f"{left} {op} {right}"
        else:
            return f"<{type(expr).__name__}>" 
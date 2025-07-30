"""
Bytecode-based parser for Python code analysis.
Used as a fallback when AST parsing fails.
"""

import dis
import inspect
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .ast_parser import FlowNode, FlowEdge


class BytecodeParser:
    """Parser that uses Python bytecode to analyze code structure."""
    
    def __init__(self):
        self.nodes: Dict[str, FlowNode] = {}
        self.edges: List[FlowEdge] = []
        self.node_counter = 0
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Python file using bytecode and return flowchart data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Compile the code to get bytecode
            code_obj = compile(source_code, file_path, 'exec')
            self._parse_bytecode(code_obj, file_path)
            
            return {
                'nodes': list(self.nodes.values()),
                'edges': self.edges,
                'source_code': source_code
            }
        except Exception as e:
            raise ValueError(f"Failed to parse file {file_path}: {str(e)}")
    
    def parse_string(self, source_code: str) -> Dict[str, Any]:
        """Parse Python code string using bytecode and return flowchart data."""
        try:
            code_obj = compile(source_code, "string_input", 'exec')
            self._parse_bytecode(code_obj, "string_input")
            
            return {
                'nodes': list(self.nodes.values()),
                'edges': self.edges,
                'source_code': source_code
            }
        except Exception as e:
            raise ValueError(f"Failed to parse source code: {str(e)}")
    
    def _parse_bytecode(self, code_obj, source_name: str):
        """Parse bytecode and build flowchart structure."""
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        
        # Add start node
        start_node = self._create_node("Start", "start", 1)
        
        # Get bytecode instructions
        instructions = list(dis.get_instructions(code_obj))
        
        # Group instructions by basic blocks
        blocks = self._group_into_blocks(instructions)
        
        # Create nodes for each block
        block_nodes = {}
        for i, block in enumerate(blocks):
            block_text = self._get_block_text(block)
            node = self._create_node(block_text, "bytecode_block", block[0].starts_line or 1)
            block_nodes[i] = node
        
        # Connect blocks
        for i, block in enumerate(blocks):
            if i > 0:
                self._create_edge(block_nodes[i-1].id, block_nodes[i].id)
        
        # Connect first block to start
        if block_nodes:
            self._create_edge(start_node.id, block_nodes[0].id)
        
        # Add end node
        if block_nodes:
            end_node = self._create_node("End", "end", max(n.line_number for n in self.nodes.values()) + 1)
            self._create_edge(block_nodes[len(blocks)-1].id, end_node.id)
    
    def _group_into_blocks(self, instructions):
        """Group bytecode instructions into basic blocks."""
        blocks = []
        current_block = []
        
        for instr in instructions:
            current_block.append(instr)
            
            # Check if this instruction ends a block
            if instr.opname in ['RETURN_VALUE', 'RAISE_VARARGS', 'BREAK_LOOP', 'CONTINUE_LOOP']:
                blocks.append(current_block)
                current_block = []
            elif instr.opname in ['JUMP_FORWARD', 'JUMP_ABSOLUTE']:
                blocks.append(current_block)
                current_block = []
            elif instr.opname in ['POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE']:
                blocks.append(current_block)
                current_block = []
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _get_block_text(self, block):
        """Get text representation of a bytecode block."""
        if not block:
            return "Empty Block"
        
        # Get the main operation
        main_op = block[0].opname
        
        # Try to get meaningful text based on operation
        if main_op == 'LOAD_CONST':
            const_value = block[0].argval
            return f"Load: {const_value}"
        elif main_op == 'LOAD_NAME':
            name = block[0].argval
            return f"Load: {name}"
        elif main_op == 'CALL_FUNCTION':
            return "Function Call"
        elif main_op == 'BINARY_ADD':
            return "Add Operation"
        elif main_op == 'COMPARE_OP':
            return "Compare Operation"
        elif main_op == 'POP_JUMP_IF_FALSE':
            return "Conditional Jump"
        elif main_op == 'POP_JUMP_IF_TRUE':
            return "Conditional Jump"
        elif main_op == 'JUMP_FORWARD':
            return "Jump Forward"
        elif main_op == 'JUMP_ABSOLUTE':
            return "Jump"
        elif main_op == 'RETURN_VALUE':
            return "Return"
        elif main_op == 'RAISE_VARARGS':
            return "Raise Exception"
        elif main_op == 'BREAK_LOOP':
            return "Break Loop"
        elif main_op == 'CONTINUE_LOOP':
            return "Continue Loop"
        else:
            return f"Operation: {main_op}"
    
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
"""
Natural language templates for Sudoku explanations
"""

from typing import Dict, List, Any, Optional


class ExplanationTemplates:
    """Templates for generating human-readable explanations"""
    
    # Templates for different techniques
    TEMPLATES = {
        "naked_single": {
            "template": "{cell_position} has only candidate {value} → Place {value} in {cell_position}.",
            "description": "A cell that has only one possible candidate"
        },
        
        "hidden_single": {
            "template": "In {unit_name}, only {cell_position} can be {value} → Place {value} in {cell_position}.",
            "description": "A digit that can only go in one cell within a unit"
        },
        
        "pointing_row": {
            "template": "In {box_name}, digit {digit} appears only in {unit_name} at {locked_positions} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit locked to one row within a box"
        },
        
        "pointing_col": {
            "template": "In {box_name}, digit {digit} appears only in {unit_name} at {locked_positions} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit locked to one column within a box"
        },
        
        "claiming_row": {
            "template": "In {unit_name}, digit {digit} appears only in {box_name} at {locked_positions} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit locked to one box within a row"
        },
        
        "claiming_col": {
            "template": "In {unit_name}, digit {digit} appears only in {box_name} at {locked_positions} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit locked to one box within a column"
        },
        
        "naked_pair": {
            "template": "In {unit_name}, cells {cell_positions} contain only candidates {digits} → eliminate {digits} from {elimination_positions}.",
            "description": "Two cells with the same two candidates"
        },
        
        "naked_triple": {
            "template": "In {unit_name}, cells {cell_positions} contain only candidates {digits} → eliminate {digits} from {elimination_positions}.",
            "description": "Three cells with the same three candidates"
        },
        
        "hidden_pair": {
            "template": "In {unit_name}, digits {digits} appear only in cells {cell_positions} → eliminate other candidates from these cells.",
            "description": "Two digits that only appear in two cells"
        },
        
        "hidden_triple": {
            "template": "In {unit_name}, digits {digits} appear only in cells {cell_positions} → eliminate other candidates from these cells.",
            "description": "Three digits that only appear in three cells"
        },
        
        "x_wing": {
            "template": "X-Wing: Digit {digit} forms a pair in {base_unit_name} {base_units} covering {cover_unit_name} {cover_units} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit that forms a rectangle pattern"
        },
        
        "swordfish": {
            "template": "Swordfish: Digit {digit} forms a triple in {base_unit_name} {base_units} covering {cover_unit_name} {cover_units} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit that forms a 3x3 pattern"
        },
        
        "jellyfish": {
            "template": "Jellyfish: Digit {digit} forms a quadruple in {base_unit_name} {base_units} covering {cover_unit_name} {cover_units} → eliminate {digit} from {elimination_positions}.",
            "description": "A digit that forms a 4x4 pattern"
        },
        
        "backtracking": {
            "template": "Backtracking: Try {value} in {cell_position} (backtrack #{backtrack_count}).",
            "description": "A guess made during search"
        },
        
        "search_start": {
            "template": "Starting search with backtracking (backtrack count: {backtrack_count})",
            "description": "Beginning of search phase"
        },
        
        "search_end": {
            "template": "Search {status} after {backtrack_count} backtracks",
            "description": "End of search phase"
        }
    }
    
    # Unit name mappings
    UNIT_NAMES = {
        "row": "Row {index}",
        "col": "Column {index}",
        "box": "Box ({box_row}-{box_row_plus_2}, {box_col}-{box_col_plus_2})"
    }
    
    @classmethod
    def get_template(cls, technique: str) -> Dict[str, str]:
        """Get template for a specific technique"""
        return cls.TEMPLATES.get(technique, {
            "template": "Applied {technique} technique.",
            "description": "Unknown technique"
        })
    
    @classmethod
    def format_unit_name(cls, unit_type: str, unit_index: int) -> str:
        """Format unit name for display"""
        if unit_type == "row":
            return cls.UNIT_NAMES["row"].format(index=unit_index + 1)
        elif unit_type == "col":
            return cls.UNIT_NAMES["col"].format(index=unit_index + 1)
        elif unit_type == "box":
            box_row = (unit_index // 3) * 3 + 1
            box_col = (unit_index % 3) * 3 + 1
            return cls.UNIT_NAMES["box"].format(
                box_row=box_row,
                box_row_plus_2=box_row + 2,
                box_col=box_col,
                box_col_plus_2=box_col + 2
            )
        else:
            return f"{unit_type.title()} {unit_index + 1}"
    
    @classmethod
    def format_positions(cls, positions: List[tuple]) -> str:
        """Format list of positions as string"""
        if not positions:
            return ""
        
        formatted = [f"R{r+1}C{c+1}" for r, c in positions]
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"
    
    @classmethod
    def format_digits(cls, digits: List[int]) -> str:
        """Format list of digits as string"""
        if not digits:
            return ""
        
        sorted_digits = sorted(digits)
        if len(sorted_digits) == 1:
            return str(sorted_digits[0])
        elif len(sorted_digits) == 2:
            return f"{sorted_digits[0]} and {sorted_digits[1]}"
        else:
            return f"{', '.join(map(str, sorted_digits[:-1]))}, and {sorted_digits[-1]}"
    
    @classmethod
    def format_eliminations(cls, eliminations: List[tuple]) -> str:
        """Format eliminations as string"""
        if not eliminations:
            return ""
        
        # Group by digit
        by_digit = {}
        for (row, col), digit in eliminations:
            if digit not in by_digit:
                by_digit[digit] = []
            by_digit[digit].append((row, col))
        
        # Format each group
        parts = []
        for digit in sorted(by_digit.keys()):
            positions = cls.format_positions(by_digit[digit])
            parts.append(f"{digit} from {positions}")
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    @classmethod
    # Build params including unit_name and status for proper template formatting
    unit_name = self.templates.format_unit_name(event.unit_type, event.unit_index)
    success_flag = event.metadata.get("success", False)
    params = {
        "technique": event.technique,
        "cell_position": event.cell_position,
        "value": event.value,
        "unit_type": event.unit_type,
        "unit_index": event.unit_index,
        "unit_name": unit_name,
        "backtrack_count": event.metadata.get("backtrack_count", 0),
        "success": success_flag,
        "status": success_flag
    }
    
    @classmethod
    def get_technique_description(cls, technique: str) -> str:
        """Get description of a technique"""
        template_info = cls.get_template(technique)
        return template_info.get("description", "Unknown technique")
    
    @classmethod
    def get_all_techniques(cls) -> List[str]:
        """Get list of all available techniques"""
        return list(cls.TEMPLATES.keys())
    
    @classmethod
    def get_technique_difficulty(cls, technique: str) -> str:
        """Get difficulty level of a technique"""
        difficulty_map = {
            "naked_single": "Easy",
            "hidden_single": "Easy",
            "pointing_row": "Medium",
            "pointing_col": "Medium",
            "claiming_row": "Medium",
            "claiming_col": "Medium",
            "naked_pair": "Medium",
            "hidden_pair": "Medium",
            "naked_triple": "Hard",
            "hidden_triple": "Hard",
            "x_wing": "Hard",
            "swordfish": "Very Hard",
            "jellyfish": "Very Hard",
            "backtracking": "Search",
            "search_start": "Search",
            "search_end": "Search"
        }
        
        return difficulty_map.get(technique, "Unknown")

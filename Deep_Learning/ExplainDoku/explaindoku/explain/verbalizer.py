"""
Verbalizer for converting trace steps into natural language explanations
"""

from typing import List, Dict, Any, Optional
from .trace import TraceEvent, SolveTrace
from .templates import ExplanationTemplates


class Verbalizer:
    """Converts trace events into natural language explanations"""
    
    def __init__(self):
        self.templates = ExplanationTemplates()
    
    def verbalize_event(self, event: TraceEvent) -> str:
        """Convert a single trace event to natural language"""
        if event.technique in self.templates.TEMPLATES:
            return self._verbalize_known_technique(event)
        else:
            return self._verbalize_unknown_technique(event)
    
    def verbalize_trace(self, trace: SolveTrace) -> List[str]:
        """Convert a complete trace to a list of explanations"""
        explanations = []
        
        for event in trace.events:
            explanation = self.verbalize_event(event)
            explanations.append(f"Step {event.step_number}: {explanation}")
        
        return explanations
    
    def verbalize_trace_summary(self, trace: SolveTrace) -> str:
        """Generate a summary of the solving trace"""
        if not trace.events:
            return "No solving steps recorded."
        
        # Count techniques
        technique_counts = {}
        for event in trace.events:
            technique = event.technique
            technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        # Generate summary
        summary_parts = []
        
        if trace.success:
            summary_parts.append("✅ Puzzle solved successfully!")
        else:
            summary_parts.append("❌ Puzzle could not be solved.")
        
        summary_parts.append(f"Total steps: {trace.total_steps}")
        summary_parts.append(f"Human strategy steps: {trace.human_steps}")
        summary_parts.append(f"Search steps: {trace.search_steps}")
        
        if trace.backtrack_count > 0:
            summary_parts.append(f"Backtracks: {trace.backtrack_count}")
        
        summary_parts.append(f"Difficulty estimate: {trace.difficulty_estimate}")
        
        # Add technique breakdown
        if technique_counts:
            summary_parts.append("\nTechniques used:")
            for technique, count in sorted(technique_counts.items()):
                difficulty = self.templates.get_technique_difficulty(technique)
                description = self.templates.get_technique_description(technique)
                summary_parts.append(f"  • {technique.replace('_', ' ').title()}: {count} times ({difficulty})")
        
        return "\n".join(summary_parts)
    
    def _verbalize_known_technique(self, event: TraceEvent) -> str:
        """Verbalize a known technique using templates"""
        params = {
            "technique": event.technique,
            "cell_position": event.cell_position,
            "value": event.value,
            "backtrack_count": event.metadata.get("backtrack_count", 0),
            "success": event.metadata.get("success", False)
        }

        # Add unit info and pre-formatted unit name if available
        if event.unit_type is not None and event.unit_index is not None:
            params["unit_type"] = event.unit_type
            params["unit_index"] = event.unit_index
            params["unit_name"] = self.templates.format_unit_name(event.unit_type, event.unit_index)

        # Add eliminations if present
        if event.eliminations:
            eliminations = []
            for elim in event.eliminations:
                eliminations.append((
                    (elim["row"], elim["col"]),
                    elim["digit"]
                ))
            params["eliminations"] = eliminations

        # Add evidence if present
        if event.evidence:
            params["evidence"] = event.evidence

        try:
            return self.templates.generate_explanation(event.technique, **params)
        except KeyError:
            # Fallback to recorded description if templates need fields we don't have
            return event.description or f"Applied {event.technique.replace('_', ' ').title()}."
    def _verbalize_unknown_technique(self, event: TraceEvent) -> str:
        """Verbalize an unknown technique"""
        base_explanation = f"Applied {event.technique} technique"
        
        if event.cell_position:
            base_explanation += f" at {event.cell_position}"
        
        if event.value:
            base_explanation += f" with value {event.value}"
        
        if event.description:
            base_explanation += f": {event.description}"
        
        return base_explanation
    
    def generate_step_by_step_explanation(self, trace: SolveTrace) -> str:
        """Generate a complete step-by-step explanation"""
        if not trace.events:
            return "No solving steps to explain."
        
        lines = []
        lines.append("🧩 **Step-by-Step Solution**")
        lines.append("=" * 50)
        
        for event in trace.events:
            explanation = self.verbalize_event(event)
            difficulty = self.templates.get_technique_difficulty(event.technique)
            
            lines.append(f"\n**Step {event.step_number}** — {event.technique.replace('_', ' ').title()} ({difficulty}):")
            lines.append(explanation)
        
        # Add summary
        lines.append("\n" + "=" * 50)
        lines.append(self.verbalize_trace_summary(trace))
        
        return "\n".join(lines)
    
    def generate_technique_explanation(self, technique: str) -> str:
        """Generate explanation of what a technique does"""
        description = self.templates.get_technique_description(technique)
        difficulty = self.templates.get_technique_difficulty(technique)
        
        return f"**{technique.replace('_', ' ').title()}** ({difficulty}): {description}"
    
    def get_technique_help(self, technique: str) -> Dict[str, Any]:
        """Get detailed help for a technique"""
        template_info = self.templates.get_template(technique)
        difficulty = self.templates.get_technique_difficulty(technique)
        
        return {
            "name": technique.replace('_', ' ').title(),
            "description": template_info.get("description", "Unknown technique"),
            "difficulty": difficulty,
            "template": template_info.get("template", ""),
            "category": self._get_technique_category(technique)
        }
    
    def _get_technique_category(self, technique: str) -> str:
        """Get the category of a technique"""
        if technique in ["naked_single", "hidden_single"]:
            return "Basic"
        elif technique in ["pointing_row", "pointing_col", "claiming_row", "claiming_col"]:
            return "Locked Candidates"
        elif technique in ["naked_pair", "hidden_pair", "naked_triple", "hidden_triple"]:
            return "Subsets"
        elif technique in ["x_wing", "swordfish", "jellyfish"]:
            return "Fish"
        elif technique in ["backtracking", "search_start", "search_end"]:
            return "Search"
        else:
            return "Other"
    
    def format_grid_state(self, grid_str: str) -> str:
        """Format grid state for display"""
        if len(grid_str) != 81:
            return "Invalid grid format"
        
        lines = []
        lines.append("Current Grid State:")
        lines.append("┌─────────┬─────────┬─────────┐")
        
        for row in range(9):
            if row > 0 and row % 3 == 0:
                lines.append("├─────────┼─────────┼─────────┤")
            
            line = "│"
            for col in range(9):
                if col > 0 and col % 3 == 0:
                    line += "│"
                
                index = row * 9 + col
                value = grid_str[index]
                if value == '0' or value == '.':
                    line += " · "
                else:
                    line += f" {value} "
            
            line += "│"
            lines.append(line)
        
        lines.append("└─────────┴─────────┴─────────┘")
        return "\n".join(lines)

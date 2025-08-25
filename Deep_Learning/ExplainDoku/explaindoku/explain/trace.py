"""
Structured proof trace for Sudoku solving
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from ..core.solver import SolveStep


@dataclass
class TraceEvent:
    """A single event in the solving trace"""
    step_number: int
    timestamp: str
    technique: str
    description: str
    cell_position: Optional[str] = None
    value: Optional[int] = None
    unit_type: Optional[str] = None
    unit_index: Optional[int] = None
    eliminations: List[Dict[str, Any]] = None
    evidence: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.eliminations is None:
            self.eliminations = []
        if self.evidence is None:
            self.evidence = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SolveTrace:
    """Complete trace of a Sudoku solve"""
    puzzle_id: str
    start_time: str
    end_time: str
    success: bool
    total_steps: int
    human_steps: int
    search_steps: int
    backtrack_count: int
    difficulty_estimate: str
    events: List[TraceEvent]
    final_grid: str
    
    def __post_init__(self):
        if self.events is None:
            self.events = []


class TraceRecorder:
    """Records solving traces for analysis and explanation"""
    
    def __init__(self, puzzle_id: str = None):
        self.puzzle_id = puzzle_id or f"puzzle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now().isoformat()
        self.events = []
        self.step_counter = 0
    
    def record_step(self, solve_step: SolveStep) -> TraceEvent:
        """Record a solving step as a trace event"""
        self.step_counter += 1
        
        event = TraceEvent(
            step_number=self.step_counter,
            timestamp=datetime.now().isoformat(),
            technique=solve_step.technique,
            description=solve_step.explanation,
            cell_position=self._format_position(solve_step.cell_position),
            value=solve_step.value,
            eliminations=self._format_eliminations(solve_step.eliminations)
        )
        
        self.events.append(event)
        return event
    
    def record_search_start(self, backtrack_count: int = 0) -> TraceEvent:
        """Record the start of search/backtracking"""
        self.step_counter += 1
        
        event = TraceEvent(
            step_number=self.step_counter,
            timestamp=datetime.now().isoformat(),
            technique="search_start",
            description=f"Starting search with backtracking (backtrack count: {backtrack_count})",
            metadata={"backtrack_count": backtrack_count}
        )
        
        self.events.append(event)
        return event
    
    def record_search_end(self, success: bool, backtrack_count: int) -> TraceEvent:
        """Record the end of search/backtracking"""
        self.step_counter += 1
        
        status = "successful" if success else "failed"
        event = TraceEvent(
            step_number=self.step_counter,
            timestamp=datetime.now().isoformat(),
            technique="search_end",
            description=f"Search {status} after {backtrack_count} backtracks",
            metadata={"success": success, "backtrack_count": backtrack_count}
        )
        
        self.events.append(event)
        return event
    
    def finalize_trace(self, success: bool, human_steps: int, search_steps: int, 
                      backtrack_count: int, difficulty_estimate: str, final_grid: str) -> SolveTrace:
        """Finalize the trace with summary information"""
        end_time = datetime.now().isoformat()
        
        return SolveTrace(
            puzzle_id=self.puzzle_id,
            start_time=self.start_time,
            end_time=end_time,
            success=success,
            total_steps=len(self.events),
            human_steps=human_steps,
            search_steps=search_steps,
            backtrack_count=backtrack_count,
            difficulty_estimate=difficulty_estimate,
            events=self.events.copy(),
            final_grid=final_grid
        )
    
    def _format_position(self, position: Optional[tuple]) -> Optional[str]:
        """Format cell position as string"""
        if position is None:
            return None
        row, col = position
        return f"R{row+1}C{col+1}"
    
    def _format_eliminations(self, eliminations: List[tuple]) -> List[Dict[str, Any]]:
        """Format eliminations as structured data"""
        formatted = []
        for (row, col), digit in eliminations:
            formatted.append({
                "position": f"R{row+1}C{col+1}",
                "row": row,
                "col": col,
                "digit": digit
            })
        return formatted
    
    def to_json(self) -> Dict[str, Any]:
        """Convert trace to JSON-serializable format"""
        return asdict(self.finalize_trace(
            success=False,  # Placeholder
            human_steps=0,
            search_steps=0,
            backtrack_count=0,
            difficulty_estimate="Unknown",
            final_grid=""
        ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the trace"""
        technique_counts = {}
        for event in self.events:
            technique = event.technique
            technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        return {
            "total_events": len(self.events),
            "technique_counts": technique_counts,
            "duration_seconds": self._calculate_duration()
        }
    
    def _calculate_duration(self) -> float:
        """Calculate duration of solving in seconds"""
        if not self.events:
            return 0.0
        
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.events[-1].timestamp)
        return (end - start).total_seconds()

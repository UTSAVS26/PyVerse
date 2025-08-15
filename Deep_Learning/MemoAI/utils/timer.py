import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GameTimer:
    """Timer class for tracking game timing and reaction times."""
    
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    move_times: List[float] = None
    last_move_time: Optional[float] = None
    
    def __post_init__(self):
        if self.move_times is None:
            self.move_times = []
    
    def start_game(self) -> None:
        """Start the game timer."""
        self.start_time = time.time()
        self.move_times = []
        self.last_move_time = self.start_time
    
    def record_move(self) -> float:
        """Record a move and return the time since last move."""
        current_time = time.time()
        if self.last_move_time is not None:
            reaction_time = current_time - self.last_move_time
            self.move_times.append(reaction_time)
        self.last_move_time = current_time
        return self.get_current_reaction_time()
    
    def end_game(self) -> float:
        """End the game timer and return total duration."""
        self.end_time = time.time()
        return self.get_total_time()
    
    def get_total_time(self) -> float:
        """Get total game time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def get_current_reaction_time(self) -> float:
        """Get the reaction time of the last move."""
        if not self.move_times:
            return 0.0
        return self.move_times[-1]
    
    def get_average_reaction_time(self) -> float:
        """Get average reaction time across all moves."""
        if not self.move_times:
            return 0.0
        return sum(self.move_times) / len(self.move_times)
    
    def get_reaction_time_stats(self) -> Dict[str, float]:
        """Get comprehensive reaction time statistics."""
        if not self.move_times:
            return {
                'avg_reaction_time': 0.0,
                'min_reaction_time': 0.0,
                'max_reaction_time': 0.0,
                'std_reaction_time': 0.0,
                'total_moves': 0
            }
        
        import statistics
        return {
            'avg_reaction_time': statistics.mean(self.move_times),
            'min_reaction_time': min(self.move_times),
            'max_reaction_time': max(self.move_times),
            'std_reaction_time': statistics.stdev(self.move_times) if len(self.move_times) > 1 else 0.0,
            'total_moves': len(self.move_times)
        }


class SessionTimer:
    """Timer for tracking multiple game sessions."""
    
    def __init__(self):
        self.sessions: List[Dict] = []
        self.current_session: Optional[Dict] = None
    
    def start_session(self, grid_size: int) -> None:
        """Start a new game session."""
        self.current_session = {
            'session_id': len(self.sessions) + 1,
            'start_time': datetime.now().isoformat(),
            'grid_size': grid_size,
            'timer': GameTimer(),
            'mistakes': 0,
            'completed': False
        }
        self.current_session['timer'].start_game()
    
    def record_move(self, is_mistake: bool = False) -> float:
        """Record a move in the current session."""
        if self.current_session is None:
            return 0.0
        
        reaction_time = self.current_session['timer'].record_move()
        if is_mistake:
            self.current_session['mistakes'] += 1
        
        return reaction_time
    
    def end_session(self) -> Dict:
        """End the current session and return session data."""
        if self.current_session is None:
            return {}
        
        self.current_session['timer'].end_game()
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['completed'] = True
        
        # Calculate session statistics
        timer = self.current_session['timer']
        stats = timer.get_reaction_time_stats()
        
        # Create session data without the timer object (which is not JSON serializable)
        session_data = {
            'session_id': self.current_session['session_id'],
            'start_time': self.current_session['start_time'],
            'end_time': self.current_session['end_time'],
            'grid_size': self.current_session['grid_size'],
            'mistakes': self.current_session['mistakes'],
            'completed': self.current_session['completed'],
            'total_time': timer.get_total_time(),
            'avg_reaction_time': stats['avg_reaction_time'],
            'min_reaction_time': stats['min_reaction_time'],
            'max_reaction_time': stats['max_reaction_time'],
            'std_reaction_time': stats['std_reaction_time'],
            'total_moves': stats['total_moves']
        }
        
        self.sessions.append(session_data)
        self.current_session = None
        
        return session_data
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all completed sessions."""
        return [session for session in self.sessions if session.get('completed', False)]
    
    def get_latest_session(self) -> Optional[Dict]:
        """Get the most recent session."""
        completed_sessions = self.get_all_sessions()
        return completed_sessions[-1] if completed_sessions else None 
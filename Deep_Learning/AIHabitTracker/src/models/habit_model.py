"""
Core habit tracking models and data structures.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class HabitEntry:
    """Represents a single day's habit entry."""
    
    date: date
    sleep_hours: float
    exercise_minutes: int
    screen_time_hours: float
    water_glasses: int
    work_hours: float
    mood_rating: int  # 1-5 scale
    productivity_rating: int  # 1-5 scale
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate the habit entry data."""
        if not 1 <= self.mood_rating <= 5:
            raise ValueError("Mood rating must be between 1 and 5")
        if not 1 <= self.productivity_rating <= 5:
            raise ValueError("Productivity rating must be between 1 and 5")
        if self.sleep_hours < 0 or self.sleep_hours > 24:
            raise ValueError("Sleep hours must be between 0 and 24")
        if self.exercise_minutes < 0:
            raise ValueError("Exercise minutes cannot be negative")
        if self.screen_time_hours < 0:
            raise ValueError("Screen time cannot be negative")
        if self.water_glasses < 0:
            raise ValueError("Water glasses cannot be negative")
        if self.work_hours < 0 or self.work_hours > 24:
            raise ValueError("Work hours must be between 0 and 24")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert habit entry to dictionary for database storage."""
        return {
            'date': self.date.isoformat(),
            'sleep_hours': self.sleep_hours,
            'exercise_minutes': self.exercise_minutes,
            'screen_time_hours': self.screen_time_hours,
            'water_glasses': self.water_glasses,
            'work_hours': self.work_hours,
            'mood_rating': self.mood_rating,
            'productivity_rating': self.productivity_rating,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HabitEntry':
        """Create habit entry from dictionary."""
        return cls(
            date=datetime.fromisoformat(data['date']).date(),
            sleep_hours=data['sleep_hours'],
            exercise_minutes=data['exercise_minutes'],
            screen_time_hours=data['screen_time_hours'],
            water_glasses=data['water_glasses'],
            work_hours=data['work_hours'],
            mood_rating=data['mood_rating'],
            productivity_rating=data['productivity_rating'],
            notes=data.get('notes')
        )


class HabitTracker:
    """Main habit tracking class that manages habit entries."""
    
    def __init__(self):
        self.entries: Dict[date, HabitEntry] = {}
    
    def add_entry(self, entry: HabitEntry) -> None:
        """Add a new habit entry."""
        self.entries[entry.date] = entry
    
    def get_entry(self, date: date) -> Optional[HabitEntry]:
        """Get habit entry for a specific date."""
        return self.entries.get(date)
    
    def update_entry(self, entry: HabitEntry) -> None:
        """Update an existing habit entry."""
        if entry.date not in self.entries:
            raise ValueError(f"No entry found for date {entry.date}")
        self.entries[entry.date] = entry
    
    def delete_entry(self, date: date) -> None:
        """Delete a habit entry for a specific date."""
        if date in self.entries:
            del self.entries[date]
    
    def get_entries_in_range(self, start_date: date, end_date: date) -> list[HabitEntry]:
        """Get all entries within a date range."""
        return [
            entry for entry in self.entries.values()
            if start_date <= entry.date <= end_date
        ]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all entries to a pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame()
        
        data = [entry.to_dict() for entry in self.entries.values()]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all entries."""
        if not self.entries:
            return {}
        
        df = self.to_dataframe()
        
        return {
            'total_entries': len(self.entries),
            'date_range': {
                'start': df['date'].min().date(),
                'end': df['date'].max().date()
            },
            'averages': {
                'sleep_hours': df['sleep_hours'].mean(),
                'exercise_minutes': df['exercise_minutes'].mean(),
                'screen_time_hours': df['screen_time_hours'].mean(),
                'water_glasses': df['water_glasses'].mean(),
                'work_hours': df['work_hours'].mean(),
                'mood_rating': df['mood_rating'].mean(),
                'productivity_rating': df['productivity_rating'].mean()
            },
            'best_days': {
                'highest_mood': df.loc[df['mood_rating'].idxmax()].to_dict(),
                'highest_productivity': df.loc[df['productivity_rating'].idxmax()].to_dict()
            }
        }

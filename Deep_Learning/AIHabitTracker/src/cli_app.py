"""
Command-line interface for AI Habit Tracker.
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from typing import Optional

from src.models.database import DatabaseManager
from src.models.habit_model import HabitEntry
from src.analysis.pattern_detector import PatternDetector
from src.analysis.visualizer import HabitVisualizer


class HabitTrackerCLI:
    """Command-line interface for habit tracking."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.db_manager = DatabaseManager()
        self.tracker = self.db_manager.load_tracker()
        self.pattern_detector = PatternDetector(self.tracker)
        self.visualizer = HabitVisualizer(self.tracker)
    
    def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(description="AI Habit Tracker CLI")
        parser.add_argument("command", choices=[
            "add", "view", "insights", "stats", "export", "clear", "interactive"
        ], help="Command to execute")
        parser.add_argument("--date", help="Date (YYYY-MM-DD)")
        parser.add_argument("--output", help="Output file for export")
        
        args = parser.parse_args()
        
        if args.command == "add":
            self.add_entry_interactive(args.date)
        elif args.command == "view":
            self.view_entries(args.date)
        elif args.command == "insights":
            self.show_insights()
        elif args.command == "stats":
            self.show_stats()
        elif args.command == "export":
            self.export_data(args.output)
        elif args.command == "clear":
            self.clear_data()
        elif args.command == "interactive":
            self.interactive_mode()
    
    def add_entry_interactive(self, entry_date: Optional[str] = None):
        """Add a habit entry interactively."""
        print("=== Add Habit Entry ===")
        
        # Get date
        if entry_date:
            try:
                date_obj = datetime.fromisoformat(entry_date).date()
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD")
                return
        else:
            date_str = input(f"Date (YYYY-MM-DD) [default: {date.today()}]: ").strip()
            if not date_str:
                date_obj = date.today()
            else:
                try:
                    date_obj = datetime.fromisoformat(date_str).date()
                except ValueError:
                    print("Invalid date format. Using today's date.")
                    date_obj = date.today()
        
        # Get habit data
        print("\nEnter your daily habits:")
        
        sleep_hours = self._get_float_input("Sleep hours", 7.0, 0.0, 24.0)
        exercise_minutes = self._get_int_input("Exercise minutes", 30, 0, 480)
        screen_time_hours = self._get_float_input("Screen time hours", 4.0, 0.0, 24.0)
        water_glasses = self._get_int_input("Water glasses", 8, 0, 50)
        work_hours = self._get_float_input("Work hours", 8.0, 0.0, 24.0)
        
        print("\nRate your day (1-5 scale):")
        mood_rating = self._get_int_input("Mood rating", 3, 1, 5)
        productivity_rating = self._get_int_input("Productivity rating", 3, 1, 5)
        
        notes = input("Notes (optional): ").strip()
        if not notes:
            notes = None
        
        # Create and save entry
        try:
            entry = HabitEntry(
                date=date_obj,
                sleep_hours=sleep_hours,
                exercise_minutes=exercise_minutes,
                screen_time_hours=screen_time_hours,
                water_glasses=water_glasses,
                work_hours=work_hours,
                mood_rating=mood_rating,
                productivity_rating=productivity_rating,
                notes=notes
            )
            
            if self.db_manager.save_entry(entry):
                self.tracker.add_entry(entry)
                print(f"✅ Entry saved for {date_obj}")
            else:
                print("❌ Failed to save entry")
                
        except ValueError as e:
            print(f"❌ Validation error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def view_entries(self, entry_date: Optional[str] = None):
        """View habit entries."""
        if entry_date:
            try:
                date_obj = datetime.fromisoformat(entry_date).date()
                entry = self.db_manager.get_entry(date_obj)
                if entry:
                    self._print_entry(entry)
                else:
                    print(f"No entry found for {date_obj}")
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD")
        else:
            entries = self.db_manager.get_all_entries()
            if entries:
                print(f"=== Recent Entries ({len(entries)} total) ===")
                for entry in entries[-10:]:  # Show last 10 entries
                    self._print_entry(entry, show_notes=False)
                    print("-" * 50)
            else:
                print("No entries found.")
    
    def show_insights(self):
        """Show AI insights."""
        print("=== AI Insights ===")
        
        insights = self.pattern_detector.generate_insights()
        recommendations = self.pattern_detector.get_recommendations()
        predictions = self.pattern_detector.predict_productivity()
        
        print("\n💡 Key Insights:")
        for insight in insights:
            print(f"  • {insight}")
        
        print("\n🎯 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
        
        print("\n🔮 Predictions:")
        if 'message' in predictions:
            print(f"  {predictions['message']}")
        else:
            print(f"  Predicted Productivity: {predictions['predicted_productivity']}/5")
            print(f"  Confidence: {predictions['confidence']:.2f}")
            print("  Feature Importance:")
            for feature, importance in predictions['feature_importance'].items():
                print(f"    - {feature}: {importance:.3f}")
    
    def show_stats(self):
        """Show summary statistics."""
        print("=== Summary Statistics ===")
        
        stats = self.db_manager.get_summary_stats()
        
        if not stats or stats.get('total_entries', 0) == 0:
            print("No data available.")
            return
        
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        
        print("\nAverages:")
        averages = stats['averages']
        print(f"  Sleep: {averages['sleep_hours']:.1f} hours")
        print(f"  Exercise: {averages['exercise_minutes']:.0f} minutes")
        print(f"  Screen Time: {averages['screen_time_hours']:.1f} hours")
        print(f"  Water: {averages['water_glasses']:.0f} glasses")
        print(f"  Work: {averages['work_hours']:.1f} hours")
        print(f"  Mood: {averages['mood_rating']:.1f}/5")
        print(f"  Productivity: {averages['productivity_rating']:.1f}/5")
        
        # Show best days
        if stats.get('best_days'):
            best_days = stats['best_days']
            if best_days.get('highest_mood'):
                print(f"\nBest Mood Day: {best_days['highest_mood']['date']} (Rating: {best_days['highest_mood']['mood_rating']})")
            if best_days.get('highest_productivity'):
                print(f"Best Productivity Day: {best_days['highest_productivity']['date']} (Rating: {best_days['highest_productivity']['productivity_rating']})")
    
    def export_data(self, output_file: Optional[str] = None):
        """Export data to CSV."""
        if not output_file:
            output_file = f"habit_data_{date.today().isoformat()}.csv"
        
        try:
            df = self.tracker.to_dataframe()
            if not df.empty:
                df.to_csv(output_file, index=False)
                print(f"✅ Data exported to {output_file}")
            else:
                print("No data to export.")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def clear_data(self):
        """Clear all data."""
        print("⚠️  This will delete ALL habit data permanently!")
        confirm = input("Are you sure? Type 'yes' to confirm: ")
        
        if confirm.lower() == 'yes':
            if self.db_manager.clear_all_data():
                self.tracker = self.db_manager.load_tracker()
                self.pattern_detector = PatternDetector(self.tracker)
                self.visualizer = HabitVisualizer(self.tracker)
                print("✅ All data cleared successfully!")
            else:
                print("❌ Failed to clear data.")
        else:
            print("Operation cancelled.")
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("=== AI Habit Tracker Interactive Mode ===")
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif command in ['help', 'h']:
                    self._show_help()
                elif command in ['add', 'a']:
                    self.add_entry_interactive()
                elif command in ['view', 'v']:
                    self.view_entries()
                elif command in ['insights', 'i']:
                    self.show_insights()
                elif command in ['stats', 's']:
                    self.show_stats()
                elif command in ['export', 'e']:
                    self.export_data()
                elif command in ['clear', 'c']:
                    self.clear_data()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _get_float_input(self, prompt: str, default: float, min_val: float, max_val: float) -> float:
        """Get float input with validation."""
        while True:
            try:
                value_str = input(f"{prompt} (default: {default}): ").strip()
                if not value_str:
                    return default
                
                value = float(value_str)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Value must be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    def _get_int_input(self, prompt: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer input with validation."""
        while True:
            try:
                value_str = input(f"{prompt} (default: {default}): ").strip()
                if not value_str:
                    return default
                
                value = int(value_str)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Value must be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid integer")
    
    def _print_entry(self, entry: HabitEntry, show_notes: bool = True):
        """Print a habit entry."""
        print(f"Date: {entry.date}")
        print(f"Sleep: {entry.sleep_hours} hours")
        print(f"Exercise: {entry.exercise_minutes} minutes")
        print(f"Screen Time: {entry.screen_time_hours} hours")
        print(f"Water: {entry.water_glasses} glasses")
        print(f"Work: {entry.work_hours} hours")
        print(f"Mood: {entry.mood_rating}/5")
        print(f"Productivity: {entry.productivity_rating}/5")
        if show_notes and entry.notes:
            print(f"Notes: {entry.notes}")
    
    def _show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("  add, a     - Add a new habit entry")
        print("  view, v    - View recent entries")
        print("  insights, i - Show AI insights")
        print("  stats, s   - Show summary statistics")
        print("  export, e  - Export data to CSV")
        print("  clear, c   - Clear all data")
        print("  help, h    - Show this help")
        print("  quit, q    - Exit the application")


def main():
    """Main function."""
    cli = HabitTrackerCLI()
    cli.run()


if __name__ == "__main__":
    main()

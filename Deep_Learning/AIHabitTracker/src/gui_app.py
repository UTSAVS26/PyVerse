"""
GUI application for AI Habit Tracker using Tkinter.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date, timedelta
import threading
import webbrowser
import os

from src.models.database import DatabaseManager
from src.models.habit_model import HabitEntry
from src.analysis.pattern_detector import PatternDetector
from src.analysis.visualizer import HabitVisualizer


class HabitTrackerGUI:
    """Main GUI application for habit tracking."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("AI Habit Tracker with Pattern Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize database and tracker
        self.db_manager = DatabaseManager()
        self.tracker = self.db_manager.load_tracker()
        self.pattern_detector = PatternDetector(self.tracker)
        self.visualizer = HabitVisualizer(self.tracker)
        
        # Create GUI elements
        self.create_widgets()
        self.update_summary()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_logging_tab()
        self.create_insights_tab()
        self.create_analytics_tab()
        self.create_data_tab()
        
    def create_logging_tab(self):
        """Create the habit logging tab."""
        logging_frame = ttk.Frame(self.notebook)
        self.notebook.add(logging_frame, text="Log Habits")
        
        # Title
        title_label = ttk.Label(logging_frame, text="Daily Habit Logger", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Main form frame
        form_frame = ttk.Frame(logging_frame)
        form_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Date selection
        date_frame = ttk.LabelFrame(form_frame, text="Date", padding=10)
        date_frame.pack(fill='x', pady=5)
        
        self.date_var = tk.StringVar(value=date.today().isoformat())
        date_entry = ttk.Entry(date_frame, textvariable=self.date_var, font=('Arial', 12))
        date_entry.pack(side='left', padx=5)
        
        today_btn = ttk.Button(date_frame, text="Today", command=self.set_today)
        today_btn.pack(side='left', padx=5)
        
        # Habit inputs
        habits_frame = ttk.LabelFrame(form_frame, text="Daily Habits", padding=10)
        habits_frame.pack(fill='x', pady=5)
        
        # Sleep hours
        sleep_frame = ttk.Frame(habits_frame)
        sleep_frame.pack(fill='x', pady=2)
        ttk.Label(sleep_frame, text="Sleep (hours):", width=15).pack(side='left')
        self.sleep_var = tk.DoubleVar(value=7.0)
        sleep_entry = ttk.Entry(sleep_frame, textvariable=self.sleep_var, width=10)
        sleep_entry.pack(side='left', padx=5)
        
        # Exercise minutes
        exercise_frame = ttk.Frame(habits_frame)
        exercise_frame.pack(fill='x', pady=2)
        ttk.Label(exercise_frame, text="Exercise (min):", width=15).pack(side='left')
        self.exercise_var = tk.IntVar(value=30)
        exercise_entry = ttk.Entry(exercise_frame, textvariable=self.exercise_var, width=10)
        exercise_entry.pack(side='left', padx=5)
        
        # Screen time
        screen_frame = ttk.Frame(habits_frame)
        screen_frame.pack(fill='x', pady=2)
        ttk.Label(screen_frame, text="Screen time (hrs):", width=15).pack(side='left')
        self.screen_var = tk.DoubleVar(value=4.0)
        screen_entry = ttk.Entry(screen_frame, textvariable=self.screen_var, width=10)
        screen_entry.pack(side='left', padx=5)
        
        # Water glasses
        water_frame = ttk.Frame(habits_frame)
        water_frame.pack(fill='x', pady=2)
        ttk.Label(water_frame, text="Water (glasses):", width=15).pack(side='left')
        self.water_var = tk.IntVar(value=8)
        water_entry = ttk.Entry(water_frame, textvariable=self.water_var, width=10)
        water_entry.pack(side='left', padx=5)
        
        # Work hours
        work_frame = ttk.Frame(habits_frame)
        work_frame.pack(fill='x', pady=2)
        ttk.Label(work_frame, text="Work hours:", width=15).pack(side='left')
        self.work_var = tk.DoubleVar(value=8.0)
        work_entry = ttk.Entry(work_frame, textvariable=self.work_var, width=10)
        work_entry.pack(side='left', padx=5)
        
        # Ratings frame
        ratings_frame = ttk.LabelFrame(form_frame, text="Daily Ratings (1-5)", padding=10)
        ratings_frame.pack(fill='x', pady=5)
        
        # Mood rating
        mood_frame = ttk.Frame(ratings_frame)
        mood_frame.pack(fill='x', pady=2)
        ttk.Label(mood_frame, text="Mood:", width=15).pack(side='left')
        self.mood_var = tk.IntVar(value=3)
        mood_scale = ttk.Scale(mood_frame, from_=1, to=5, variable=self.mood_var, orient='horizontal')
        mood_scale.pack(side='left', fill='x', expand=True, padx=5)
        mood_label = ttk.Label(mood_frame, textvariable=self.mood_var)
        mood_label.pack(side='left', padx=5)
        
        # Productivity rating
        productivity_frame = ttk.Frame(ratings_frame)
        productivity_frame.pack(fill='x', pady=2)
        ttk.Label(productivity_frame, text="Productivity:", width=15).pack(side='left')
        self.productivity_var = tk.IntVar(value=3)
        productivity_scale = ttk.Scale(productivity_frame, from_=1, to=5, variable=self.productivity_var, orient='horizontal')
        productivity_scale.pack(side='left', fill='x', expand=True, padx=5)
        productivity_label = ttk.Label(productivity_frame, textvariable=self.productivity_var)
        productivity_label.pack(side='left', padx=5)
        
        # Notes
        notes_frame = ttk.LabelFrame(form_frame, text="Notes (Optional)", padding=10)
        notes_frame.pack(fill='x', pady=5)
        self.notes_text = tk.Text(notes_frame, height=3, width=50)
        self.notes_text.pack(fill='x')
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.pack(pady=10)
        
        save_btn = ttk.Button(button_frame, text="Save Entry", command=self.save_entry)
        save_btn.pack(side='left', padx=5)
        
        load_btn = ttk.Button(button_frame, text="Load Entry", command=self.load_entry)
        load_btn.pack(side='left', padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear Form", command=self.clear_form)
        clear_btn.pack(side='left', padx=5)
        
    def create_insights_tab(self):
        """Create the insights tab."""
        insights_frame = ttk.Frame(self.notebook)
        self.notebook.add(insights_frame, text="AI Insights")
        
        # Title
        title_label = ttk.Label(insights_frame, text="AI-Powered Insights & Recommendations", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(insights_frame)
        scrollbar = ttk.Scrollbar(insights_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Insights sections
        self.create_insights_section(scrollable_frame, "Key Insights", self.get_insights)
        self.create_insights_section(scrollable_frame, "Recommendations", self.get_recommendations)
        self.create_insights_section(scrollable_frame, "Pattern Analysis", self.get_patterns)
        self.create_insights_section(scrollable_frame, "Predictions", self.get_predictions)
        
        # Refresh button
        refresh_btn = ttk.Button(insights_frame, text="Refresh Insights", command=self.refresh_insights)
        refresh_btn.pack(pady=10)
        
    def create_insights_section(self, parent, title, get_data_func):
        """Create a section for displaying insights."""
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill='x', pady=5, padx=10)
        
        text_widget = tk.Text(section_frame, height=6, width=80, wrap='word')
        text_widget.pack(fill='x')
        
        # Store reference to update later
        setattr(self, f"{title.lower().replace(' ', '_')}_text", text_widget)
        setattr(self, f"{title.lower().replace(' ', '_')}_func", get_data_func)
        
    def create_analytics_tab(self):
        """Create the analytics tab."""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="Analytics")
        
        # Title
        title_label = ttk.Label(analytics_frame, text="Data Analytics & Visualizations", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(analytics_frame)
        buttons_frame.pack(pady=10)
        
        # Chart generation buttons
        ttk.Button(buttons_frame, text="Generate Dashboard", 
                  command=self.generate_dashboard).pack(pady=5)
        ttk.Button(buttons_frame, text="Correlation Heatmap", 
                  command=self.generate_correlation_heatmap).pack(pady=5)
        ttk.Button(buttons_frame, text="Trend Analysis", 
                  command=self.generate_trend_analysis).pack(pady=5)
        ttk.Button(buttons_frame, text="Weekly Summary", 
                  command=self.generate_weekly_summary).pack(pady=5)
        ttk.Button(buttons_frame, text="Save All Charts", 
                  command=self.save_all_charts).pack(pady=5)
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(analytics_frame, text="Summary Statistics", padding=10)
        summary_frame.pack(fill='x', pady=10, padx=10)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=80)
        self.summary_text.pack(fill='x')
        
    def create_data_tab(self):
        """Create the data management tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Management")
        
        # Title
        title_label = ttk.Label(data_frame, text="Data Management", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Data operations frame
        operations_frame = ttk.LabelFrame(data_frame, text="Data Operations", padding=10)
        operations_frame.pack(fill='x', pady=10, padx=10)
        
        ttk.Button(operations_frame, text="Export Data (CSV)", 
                  command=self.export_data).pack(pady=5)
        ttk.Button(operations_frame, text="Import Data (CSV)", 
                  command=self.import_data).pack(pady=5)
        ttk.Button(operations_frame, text="Clear All Data", 
                  command=self.clear_all_data).pack(pady=5)
        
        # Recent entries frame
        entries_frame = ttk.LabelFrame(data_frame, text="Recent Entries", padding=10)
        entries_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        # Create treeview for entries
        columns = ('Date', 'Sleep', 'Exercise', 'Screen', 'Water', 'Work', 'Mood', 'Productivity')
        self.entries_tree = ttk.Treeview(entries_frame, columns=columns, show='headings')
        
        for col in columns:
            self.entries_tree.heading(col, text=col)
            self.entries_tree.column(col, width=100)
        
        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(entries_frame, orient='vertical', command=self.entries_tree.yview)
        self.entries_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.entries_tree.pack(side='left', fill='both', expand=True)
        tree_scrollbar.pack(side='right', fill='y')
        
        # Bind double-click to load entry
        self.entries_tree.bind('<Double-1>', self.load_entry_from_tree)
        
        # Refresh button
        refresh_btn = ttk.Button(data_frame, text="Refresh Entries", command=self.refresh_entries)
        refresh_btn.pack(pady=10)
        
    def set_today(self):
        """Set the date to today."""
        self.date_var.set(date.today().isoformat())
        
    def save_entry(self):
        """Save the current habit entry."""
        try:
            # Parse date
            entry_date = datetime.fromisoformat(self.date_var.get()).date()
            
            # Create habit entry
            entry = HabitEntry(
                date=entry_date,
                sleep_hours=self.sleep_var.get(),
                exercise_minutes=self.exercise_var.get(),
                screen_time_hours=self.screen_var.get(),
                water_glasses=self.water_var.get(),
                work_hours=self.work_var.get(),
                mood_rating=self.mood_var.get(),
                productivity_rating=self.productivity_var.get(),
                notes=self.notes_text.get("1.0", tk.END).strip()
            )
            
            # Save to database
            if self.db_manager.save_entry(entry):
                self.tracker.add_entry(entry)
                messagebox.showinfo("Success", "Habit entry saved successfully!")
                self.update_summary()
                self.refresh_entries()
            else:
                messagebox.showerror("Error", "Failed to save entry!")
                
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def load_entry(self):
        """Load an entry for the selected date."""
        try:
            entry_date = datetime.fromisoformat(self.date_var.get()).date()
            entry = self.db_manager.get_entry(entry_date)
            
            if entry:
                self.populate_form(entry)
                messagebox.showinfo("Success", "Entry loaded successfully!")
            else:
                messagebox.showinfo("Info", "No entry found for this date.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def populate_form(self, entry):
        """Populate the form with entry data."""
        self.date_var.set(entry.date.isoformat())
        self.sleep_var.set(entry.sleep_hours)
        self.exercise_var.set(entry.exercise_minutes)
        self.screen_var.set(entry.screen_time_hours)
        self.water_var.set(entry.water_glasses)
        self.work_var.set(entry.work_hours)
        self.mood_var.set(entry.mood_rating)
        self.productivity_var.set(entry.productivity_rating)
        
        self.notes_text.delete("1.0", tk.END)
        if entry.notes:
            self.notes_text.insert("1.0", entry.notes)
            
    def clear_form(self):
        """Clear the form."""
        self.set_today()
        self.sleep_var.set(7.0)
        self.exercise_var.set(30)
        self.screen_var.set(4.0)
        self.water_var.set(8)
        self.work_var.set(8.0)
        self.mood_var.set(3)
        self.productivity_var.set(3)
        self.notes_text.delete("1.0", tk.END)
        
    def update_summary(self):
        """Update the summary statistics."""
        stats = self.db_manager.get_summary_stats()
        
        if hasattr(self, 'summary_text'):
            self.summary_text.delete("1.0", tk.END)
            
            if stats.get('total_entries', 0) == 0:
                self.summary_text.insert("1.0", "No data available. Start logging your habits!")
                return
                
            summary = f"Total Entries: {stats['total_entries']}\n"
            summary += f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}\n\n"
            
            summary += "Averages:\n"
            averages = stats['averages']
            summary += f"  Sleep: {averages['sleep_hours']:.1f} hours\n"
            summary += f"  Exercise: {averages['exercise_minutes']:.0f} minutes\n"
            summary += f"  Screen Time: {averages['screen_time_hours']:.1f} hours\n"
            summary += f"  Water: {averages['water_glasses']:.0f} glasses\n"
            summary += f"  Work: {averages['work_hours']:.1f} hours\n"
            summary += f"  Mood: {averages['mood_rating']:.1f}/5\n"
            summary += f"  Productivity: {averages['productivity_rating']:.1f}/5\n"
            
            self.summary_text.insert("1.0", summary)
            
    def refresh_insights(self):
        """Refresh all insights."""
        self.update_insights_section("key_insights", self.get_insights)
        self.update_insights_section("recommendations", self.get_recommendations)
        self.update_insights_section("pattern_analysis", self.get_patterns)
        self.update_insights_section("predictions", self.get_predictions)
        
    def update_insights_section(self, section_name, get_data_func):
        """Update a specific insights section."""
        text_widget = getattr(self, f"{section_name}_text", None)
        if text_widget:
            text_widget.delete("1.0", tk.END)
            data = get_data_func()
            if isinstance(data, list):
                text_widget.insert("1.0", "\n".join(data))
            else:
                text_widget.insert("1.0", str(data))
                
    def get_insights(self):
        """Get key insights."""
        return self.pattern_detector.generate_insights()
        
    def get_recommendations(self):
        """Get recommendations."""
        return self.pattern_detector.get_recommendations()
        
    def get_patterns(self):
        """Get pattern analysis."""
        patterns = self.pattern_detector.detect_patterns()
        if isinstance(patterns, dict) and 'message' in patterns:
            return [patterns['message']]
            
        pattern_text = []
        for pattern_type, pattern_data in patterns.items():
            if pattern_data:
                pattern_text.append(f"{pattern_type.title()} Patterns:")
                if isinstance(pattern_data, dict):
                    for key, value in pattern_data.items():
                        pattern_text.append(f"  {key}: {value}")
                pattern_text.append("")
                
        return pattern_text if pattern_text else ["No significant patterns detected yet."]
        
    def get_predictions(self):
        """Get predictions."""
        predictions = self.pattern_detector.predict_productivity()
        if 'message' in predictions:
            return [predictions['message']]
            
        pred_text = []
        pred_text.append(f"Predicted Productivity: {predictions['predicted_productivity']}/5")
        pred_text.append(f"Confidence: {predictions['confidence']:.2f}")
        pred_text.append("\nFeature Importance:")
        for feature, importance in predictions['feature_importance'].items():
            pred_text.append(f"  {feature}: {importance:.3f}")
            
        return pred_text
        
    def generate_dashboard(self):
        """Generate and open the dashboard."""
        def generate():
            try:
                fig = self.visualizer.create_dashboard()
                fig.write_html("dashboard.html")
                webbrowser.open("dashboard.html")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate dashboard: {str(e)}")
                
        threading.Thread(target=generate).start()
        
    def generate_correlation_heatmap(self):
        """Generate and open correlation heatmap."""
        def generate():
            try:
                fig = self.visualizer.create_correlation_heatmap()
                fig.write_html("correlation_heatmap.html")
                webbrowser.open("correlation_heatmap.html")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate heatmap: {str(e)}")
                
        threading.Thread(target=generate).start()
        
    def generate_trend_analysis(self):
        """Generate and open trend analysis."""
        def generate():
            try:
                fig = self.visualizer.create_trend_analysis()
                fig.write_html("trend_analysis.html")
                webbrowser.open("trend_analysis.html")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate trend analysis: {str(e)}")
                
        threading.Thread(target=generate).start()
        
    def generate_weekly_summary(self):
        """Generate and open weekly summary."""
        def generate():
            try:
                fig = self.visualizer.create_weekly_summary()
                fig.write_html("weekly_summary.html")
                webbrowser.open("weekly_summary.html")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate weekly summary: {str(e)}")
                
        threading.Thread(target=generate).start()
        
    def save_all_charts(self):
        """Save all charts to a directory."""
        def save():
            try:
                output_dir = filedialog.askdirectory(title="Select directory to save charts")
                if output_dir:
                    saved_files = self.visualizer.save_all_charts(output_dir)
                    messagebox.showinfo("Success", f"Charts saved to {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save charts: {str(e)}")
                
        threading.Thread(target=save).start()
        
    def export_data(self):
        """Export data to CSV."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                df = self.tracker.to_dataframe()
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
            
    def import_data(self):
        """Import data from CSV."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                # This would need to be implemented based on CSV structure
                messagebox.showinfo("Info", "Import functionality to be implemented")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            
    def clear_all_data(self):
        """Clear all data after confirmation."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all data? This cannot be undone."):
            if self.db_manager.clear_all_data():
                self.tracker = self.db_manager.load_tracker()
                self.pattern_detector = PatternDetector(self.tracker)
                self.visualizer = HabitVisualizer(self.tracker)
                self.update_summary()
                self.refresh_entries()
                messagebox.showinfo("Success", "All data cleared successfully!")
            else:
                messagebox.showerror("Error", "Failed to clear data!")
                
    def refresh_entries(self):
        """Refresh the entries treeview."""
        # Clear existing entries
        for item in self.entries_tree.get_children():
            self.entries_tree.delete(item)
            
        # Load entries from database
        entries = self.db_manager.get_all_entries()
        
        # Add to treeview
        for entry in entries[-20:]:  # Show last 20 entries
            self.entries_tree.insert('', 'end', values=(
                entry.date.isoformat(),
                f"{entry.sleep_hours:.1f}",
                entry.exercise_minutes,
                f"{entry.screen_time_hours:.1f}",
                entry.water_glasses,
                f"{entry.work_hours:.1f}",
                entry.mood_rating,
                entry.productivity_rating
            ))
            
    def load_entry_from_tree(self, event):
        """Load entry from treeview selection."""
        selection = self.entries_tree.selection()
        if selection:
            item = self.entries_tree.item(selection[0])
            date_str = item['values'][0]
            self.date_var.set(date_str)
            self.load_entry()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = HabitTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

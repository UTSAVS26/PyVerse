"""
KeyMentor - User Interface
CLI and GUI for user interaction and typing exercises.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from typing import Optional, Callable
import sys
import os

from tracker import TypingTracker, create_tracker
from analyzer import TypingAnalyzer, create_analyzer
from exercise_generator import ExerciseGenerator, create_exercise_generator, TypingExercise


class KeyMentorUI:
    """Main user interface for KeyMentor"""
    
    def __init__(self):
        self.tracker = create_tracker()
        self.analyzer = create_analyzer()
        self.generator = create_exercise_generator()
        self.current_exercise: Optional[TypingExercise] = None
        self.is_typing_session_active = False
        self.session_start_time: Optional[float] = None
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("KeyMentor - Typing Coach")
        self.root.geometry("800x600")
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_typing_tab()
        self.create_analysis_tab()
        self.create_exercises_tab()
        self.create_progress_tab()
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Welcome message
        welcome_label = ttk.Label(dashboard_frame, 
                                text="Welcome to KeyMentor!", 
                                font=("Arial", 16, "bold"))
        welcome_label.pack(pady=20)
        
        # Quick stats
        stats_frame = ttk.LabelFrame(dashboard_frame, text="Quick Stats")
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=60)
        self.stats_text.pack(padx=10, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(dashboard_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Start Typing Session", 
                  command=self.start_typing_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Analysis", 
                  command=lambda: self.notebook.select(2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Practice Exercises", 
                  command=lambda: self.notebook.select(3)).pack(side=tk.LEFT, padx=5)
        
        # Update stats
        self.update_dashboard_stats()
    
    def create_typing_tab(self):
        """Create the typing practice tab"""
        typing_frame = ttk.Frame(self.notebook)
        self.notebook.add(typing_frame, text="Typing Practice")
        
        # Instructions
        instructions = ttk.Label(typing_frame, 
                               text="Type the text below. Focus on accuracy and speed.",
                               font=("Arial", 12))
        instructions.pack(pady=10)
        
        # Text to type
        text_frame = ttk.LabelFrame(typing_frame, text="Text to Type")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_to_type = scrolledtext.ScrolledText(text_frame, height=8, width=60)
        self.text_to_type.pack(padx=10, pady=10)
        
        # User input
        input_frame = ttk.LabelFrame(typing_frame, text="Your Typing")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.user_input = scrolledtext.ScrolledText(input_frame, height=8, width=60)
        self.user_input.pack(padx=10, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(typing_frame)
        control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Session", 
                                     command=self.start_typing_session)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.end_button = ttk.Button(control_frame, text="End Session", 
                                   command=self.end_typing_session, state=tk.DISABLED)
        self.end_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Load Sample Text", 
                  command=self.load_sample_text).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(typing_frame, text="Session Results")
        results_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=4, width=60)
        self.results_text.pack(padx=10, pady=10)
    
    def create_analysis_tab(self):
        """Create the analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Analysis controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Analyze My Typing", 
                  command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Progress Report", 
                  command=self.generate_progress_report).pack(side=tk.LEFT, padx=5)
        
        # Analysis results
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=20, width=80)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def create_exercises_tab(self):
        """Create the exercises tab"""
        exercises_frame = ttk.Frame(self.notebook)
        self.notebook.add(exercises_frame, text="Exercises")
        
        # Exercise controls
        control_frame = ttk.Frame(exercises_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Generate Personalized Exercises", 
                  command=self.generate_exercises).pack(side=tk.LEFT, padx=5)
        
        # Difficulty selector
        ttk.Label(control_frame, text="Difficulty:").pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.StringVar(value="medium")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var,
                                      values=["easy", "medium", "hard", "expert"])
        difficulty_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Progressive Exercises", 
                  command=self.generate_progressive_exercises).pack(side=tk.LEFT, padx=5)
        
        # Exercise display
        exercise_display_frame = ttk.LabelFrame(exercises_frame, text="Generated Exercises")
        exercise_display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.exercise_text = scrolledtext.ScrolledText(exercise_display_frame, height=20, width=80)
        self.exercise_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_progress_tab(self):
        """Create the progress tracking tab"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="Progress")
        
        # Progress controls
        control_frame = ttk.Frame(progress_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="View Recent Sessions", 
                  command=self.view_recent_sessions).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Progress display
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=20, width=80)
        self.progress_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def update_dashboard_stats(self):
        """Update dashboard statistics"""
        try:
            sessions = self.tracker.get_session_history(5)
            if sessions:
                avg_wpm = sum(s.wpm for s in sessions) / len(sessions)
                avg_accuracy = sum(s.accuracy for s in sessions) / len(sessions)
                
                stats = f"Recent Sessions: {len(sessions)}\n"
                stats += f"Average WPM: {avg_wpm:.2f}\n"
                stats += f"Average Accuracy: {avg_accuracy:.2f}%\n"
                stats += f"Total Keys Typed: {sum(s.total_keys for s in sessions)}\n"
                stats += f"Last Session: {time.strftime('%Y-%m-%d %H:%M', time.localtime(sessions[0].start_time))}\n"
            else:
                stats = "No typing sessions found.\nStart your first session to see statistics!"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error loading stats: {e}")
    
    def start_typing_session(self):
        """Start a new typing session"""
        if self.is_typing_session_active:
            messagebox.showwarning("Session Active", "A typing session is already active!")
            return
        
        # Get text to type
        text = self.text_to_type.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to type first!")
            return
        
        # Start session
        self.session_start_time = time.time()
        self.tracker.start_session()
        self.is_typing_session_active = True
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.end_button.config(state=tk.NORMAL)
        self.user_input.delete(1.0, tk.END)
        self.user_input.focus()
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Session started. Begin typing...")
        
        # Bind typing events
        self.user_input.bind('<Key>', self.on_keypress)
    
    def on_keypress(self, event):
        """Handle keypress events during typing session"""
        if not self.is_typing_session_active:
            return
        
        # Get current text content
        current_text = self.user_input.get(1.0, tk.END).strip()
        expected_text = self.text_to_type.get(1.0, tk.END).strip()
        
        # Record the keypress
        if event.char and len(event.char) == 1:
            expected_char = expected_text[len(current_text)] if len(current_text) < len(expected_text) else event.char
            self.tracker.record_keypress(event.char, expected_char, current_text)
    
    def end_typing_session(self):
        """End the current typing session"""
        if not self.is_typing_session_active:
            return
        
        # End session
        session = self.tracker.end_session()
        self.is_typing_session_active = False
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.end_button.config(state=tk.DISABLED)
        self.user_input.unbind('<Key>')
        
        # Display results
        results = f"Session completed!\n"
        results += f"WPM: {session.wpm:.2f}\n"
        results += f"Accuracy: {session.accuracy:.2f}%\n"
        results += f"Total Keys: {session.total_keys}\n"
        results += f"Correct Keys: {session.correct_keys}\n"
        results += f"Mistakes: {len(session.mistakes)}"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        
        # Update dashboard
        self.update_dashboard_stats()
        
        messagebox.showinfo("Session Complete", f"Great job! Your WPM: {session.wpm:.2f}")
    
    def load_sample_text(self):
        """Load sample text for typing practice"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
            "Programming is the art of telling another human being what one wants the computer to do. It requires clear thinking and precise communication.",
            "Success is not final, failure is not fatal: it is the courage to continue that counts. Every day brings new opportunities to improve.",
            "The best way to predict the future is to invent it. Innovation comes from combining existing ideas in new and creative ways.",
            "Practice makes perfect, but perfect practice makes perfect performance. Focus on accuracy first, then speed will follow naturally."
        ]
        
        selected_text = sample_texts[hash(time.time()) % len(sample_texts)]
        self.text_to_type.delete(1.0, tk.END)
        self.text_to_type.insert(1.0, selected_text)
    
    def run_analysis(self):
        """Run typing analysis"""
        try:
            profile = self.analyzer.analyze_user_typing()
            
            analysis = f"Typing Profile Analysis\n"
            analysis += f"=" * 50 + "\n\n"
            analysis += f"Total Sessions: {profile.total_sessions}\n"
            analysis += f"Average WPM: {profile.avg_wpm:.2f}\n"
            analysis += f"Average Accuracy: {profile.avg_accuracy:.2f}%\n\n"
            
            analysis += f"Top Weak Spots:\n"
            for i, weak_spot in enumerate(profile.weak_spots[:10], 1):
                analysis += f"{i}. {weak_spot.pattern} ({weak_spot.pattern_type}) - "
                analysis += f"Error Rate: {weak_spot.error_rate:.2%}, "
                analysis += f"Difficulty: {weak_spot.difficulty_score:.3f}\n"
            
            analysis += f"\nCommon Mistakes:\n"
            for expected, mistakes in list(profile.common_mistakes.items())[:10]:
                analysis += f"{expected} -> {mistakes}\n"
            
            analysis += f"\nFinger Weaknesses:\n"
            for finger, error_rate in profile.finger_weaknesses.items():
                analysis += f"{finger}: {error_rate:.2%}\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, analysis)
            
        except ValueError as e:
            messagebox.showwarning("No Data", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
    
    def generate_progress_report(self):
        """Generate progress report"""
        try:
            progress = self.analyzer.get_progress_report(days=7)
            
            if "error" in progress:
                messagebox.showwarning("No Data", progress["error"])
                return
            
            report = f"7-Day Progress Report\n"
            report += f"=" * 30 + "\n\n"
            report += f"Total Sessions: {progress['total_sessions']}\n"
            report += f"Average WPM: {progress['avg_wpm']:.2f}\n"
            report += f"Max WPM: {progress['max_wpm']:.2f}\n"
            report += f"Min WPM: {progress['min_wpm']:.2f}\n"
            report += f"Average Accuracy: {progress['avg_accuracy']:.2f}%\n"
            report += f"Overall Accuracy: {progress['overall_accuracy']:.2f}%\n"
            report += f"WPM Trend: {progress['wpm_trend']}\n"
            report += f"Accuracy Trend: {progress['accuracy_trend']}\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, report)
            
        except Exception as e:
            messagebox.showerror("Error", f"Progress report failed: {e}")
    
    def generate_exercises(self):
        """Generate personalized exercises"""
        try:
            profile = self.analyzer.analyze_user_typing()
            exercises = self.generator.generate_exercises(profile, 5)
            
            exercise_text = f"Personalized Typing Exercises\n"
            exercise_text += f"=" * 40 + "\n\n"
            
            for i, exercise in enumerate(exercises, 1):
                exercise_text += f"Exercise {i}:\n"
                exercise_text += f"Type: {exercise.exercise_type}\n"
                exercise_text += f"Difficulty: {exercise.difficulty:.2f}\n"
                exercise_text += f"Target Patterns: {exercise.target_patterns}\n"
                exercise_text += f"Instructions: {exercise.instructions}\n"
                exercise_text += f"Text: {exercise.text}\n"
                exercise_text += f"{'='*40}\n\n"
            
            self.exercise_text.delete(1.0, tk.END)
            self.exercise_text.insert(1.0, exercise_text)
            
        except ValueError as e:
            messagebox.showwarning("No Data", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Exercise generation failed: {e}")
    
    def generate_progressive_exercises(self):
        """Generate progressive exercises"""
        try:
            profile = self.analyzer.analyze_user_typing()
            difficulty = self.difficulty_var.get()
            exercises = self.generator.generate_progressive_exercises(profile, difficulty)
            
            exercise_text = f"Progressive Exercises ({difficulty.title()})\n"
            exercise_text += f"=" * 40 + "\n\n"
            
            for i, exercise in enumerate(exercises, 1):
                exercise_text += f"Exercise {i}:\n"
                exercise_text += f"Type: {exercise.exercise_type}\n"
                exercise_text += f"Difficulty: {exercise.difficulty:.2f}\n"
                exercise_text += f"Instructions: {exercise.instructions}\n"
                exercise_text += f"Text: {exercise.text}\n"
                exercise_text += f"{'='*40}\n\n"
            
            self.exercise_text.delete(1.0, tk.END)
            self.exercise_text.insert(1.0, exercise_text)
            
        except ValueError as e:
            messagebox.showwarning("No Data", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Progressive exercise generation failed: {e}")
    
    def view_recent_sessions(self):
        """View recent typing sessions"""
        try:
            sessions = self.tracker.get_session_history(10)
            
            sessions_text = f"Recent Typing Sessions\n"
            sessions_text += f"=" * 30 + "\n\n"
            
            for i, session in enumerate(sessions, 1):
                sessions_text += f"Session {i}:\n"
                sessions_text += f"Date: {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.start_time))}\n"
                sessions_text += f"WPM: {session.wpm:.2f}\n"
                sessions_text += f"Accuracy: {session.accuracy:.2f}%\n"
                sessions_text += f"Total Keys: {session.total_keys}\n"
                sessions_text += f"Duration: {session.end_time - session.start_time:.1f}s\n"
                sessions_text += f"{'='*30}\n\n"
            
            self.progress_text.delete(1.0, tk.END)
            self.progress_text.insert(1.0, sessions_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sessions: {e}")
    
    def export_data(self):
        """Export typing data"""
        try:
            sessions = self.tracker.get_session_history(100)
            
            # Create export directory
            os.makedirs("exports", exist_ok=True)
            
            # Export to CSV
            import csv
            with open("exports/typing_sessions.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Session ID", "Start Time", "End Time", "WPM", "Accuracy", "Total Keys", "Correct Keys"])
                
                for session in sessions:
                    writer.writerow([
                        session.session_id,
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.start_time)),
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.end_time)),
                        session.wpm,
                        session.accuracy,
                        session.total_keys,
                        session.correct_keys
                    ])
            
            messagebox.showinfo("Export Complete", "Data exported to exports/typing_sessions.csv")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def run_cli():
    """Run KeyMentor in CLI mode"""
    print("KeyMentor - Typing Coach (CLI Mode)")
    print("=" * 40)
    
    tracker = create_tracker()
    analyzer = create_analyzer()
    generator = create_exercise_generator()
    
    while True:
        print("\nOptions:")
        print("1. Start typing session")
        print("2. View analysis")
        print("3. Generate exercises")
        print("4. View progress")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Start typing session
            text = input("Enter text to type (or press Enter for sample): ").strip()
            if not text:
                text = "The quick brown fox jumps over the lazy dog."
            
            print(f"\nType this text:\n{text}\n")
            input("Press Enter when ready to start...")
            
            session_id = tracker.start_session()
            user_input = input("Start typing: ")
            
            # Simulate typing events
            for i, char in enumerate(user_input):
                expected_char = text[i] if i < len(text) else char
                tracker.record_keypress(char, expected_char, user_input[:i+1])
            
            session = tracker.end_session()
            print(f"\nSession completed!")
            print(f"WPM: {session.wpm:.2f}")
            print(f"Accuracy: {session.accuracy:.2f}%")
            print(f"Total keys: {session.total_keys}")
        
        elif choice == "2":
            # View analysis
            try:
                profile = analyzer.analyze_user_typing()
                print(f"\nTyping Analysis:")
                print(f"Total Sessions: {profile.total_sessions}")
                print(f"Average WPM: {profile.avg_wpm:.2f}")
                print(f"Average Accuracy: {profile.avg_accuracy:.2f}%")
                
                print(f"\nTop Weak Spots:")
                for i, weak_spot in enumerate(profile.weak_spots[:5], 1):
                    print(f"{i}. {weak_spot.pattern} - Error Rate: {weak_spot.error_rate:.2%}")
                    
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            # Generate exercises
            try:
                profile = analyzer.analyze_user_typing()
                exercises = generator.generate_exercises(profile, 3)
                
                print(f"\nGenerated Exercises:")
                for i, exercise in enumerate(exercises, 1):
                    print(f"\nExercise {i}:")
                    print(f"Text: {exercise.text}")
                    print(f"Instructions: {exercise.instructions}")
                    
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == "4":
            # View progress
            try:
                progress = analyzer.get_progress_report(days=7)
                print(f"\n7-Day Progress Report:")
                print(f"Average WPM: {progress['avg_wpm']:.2f}")
                print(f"WPM Trend: {progress['wpm_trend']}")
                print(f"Accuracy Trend: {progress['accuracy_trend']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        app = KeyMentorUI()
        app.run()

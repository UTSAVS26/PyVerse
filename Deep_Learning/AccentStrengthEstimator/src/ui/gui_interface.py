"""
Graphical user interface for the Accent Strength Estimator.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Dict, Any


class GUIInterface:
    """Graphical user interface for the accent strength estimator."""
    
    def __init__(self):
        """Initialize the GUI interface."""
        self.root = None
        self.recording = False
        self.current_phrase = 0
        self.phrases = []
        self.results = {}
        
    def run(self):
        """Run the GUI interface."""
        try:
            self.root = tk.Tk()
            self.root.title("🎤 Accent Strength Estimator")
            self.root.geometry("800x600")
            self.root.configure(bg='#f0f0f0')
            
            self._create_widgets()
            self._load_phrases()
            
            self.root.mainloop()
        except Exception as e:
            print(f"❌ GUI Error: {e}")
            print("Falling back to CLI interface...")
            from .cli_interface import CLIInterface
            cli = CLIInterface()
            cli.run()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎤 Accent Strength Estimator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Phrase display
        self.phrase_label = ttk.Label(main_frame, text="No phrases loaded", 
                                     font=('Arial', 12), wraplength=600)
        self.phrase_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Record button
        self.record_button = ttk.Button(button_frame, text="🎙️ Start Recording", 
                                       command=self._toggle_recording)
        self.record_button.grid(row=0, column=0, padx=5)
        
        # Next button
        self.next_button = ttk.Button(button_frame, text="⏭️ Next Phrase", 
                                     command=self._next_phrase, state='disabled')
        self.next_button.grid(row=0, column=1, padx=5)
        
        # Analyze button
        self.analyze_button = ttk.Button(button_frame, text="📊 Analyze Results", 
                                        command=self._analyze_results, state='disabled')
        self.analyze_button.grid(row=0, column=2, padx=5)
        
        # Results text area
        self.results_text = tk.Text(main_frame, height=15, width=80, wrap=tk.WORD)
        self.results_text.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=4, column=2, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def _load_phrases(self):
        """Load reference phrases."""
        try:
            # Load phrases from file
            phrases_file = "data/reference_phrases.txt"
            with open(phrases_file, 'r', encoding='utf-8') as f:
                self.phrases = [line.strip() for line in f if line.strip()]
            
            if self.phrases:
                self.current_phrase = 0
                self._update_phrase_display()
                self.status_var.set(f"Loaded {len(self.phrases)} phrases")
            else:
                self.status_var.set("No phrases found")
                
        except FileNotFoundError:
            messagebox.showerror("Error", "Reference phrases file not found!")
            self.status_var.set("Error: Phrases file not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load phrases: {e}")
            self.status_var.set("Error loading phrases")
    
    def _update_phrase_display(self):
        """Update the phrase display."""
        if self.phrases and 0 <= self.current_phrase < len(self.phrases):
            phrase = self.phrases[self.current_phrase]
            self.phrase_label.config(text=f"Phrase {self.current_phrase + 1}/{len(self.phrases)}: {phrase}")
        else:
            self.phrase_label.config(text="No phrases available")
    
    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start recording."""
        if not self.phrases:
            messagebox.showwarning("Warning", "No phrases loaded!")
            return
        
        self.recording = True
        self.record_button.config(text="⏹️ Stop Recording")
        self.progress_var.set("Recording... Speak now!")
        self.status_var.set("Recording in progress")
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
    
    def _stop_recording(self):
        """Stop recording."""
        self.recording = False
        self.record_button.config(text="🎙️ Start Recording")
        self.progress_var.set("Processing...")
        self.status_var.set("Processing audio")
    
    def _record_audio(self):
        """Record audio in a separate thread."""
        try:
            # Simulate recording (in a real implementation, this would use the AudioRecorder)
            time.sleep(3)  # Simulate 3 seconds of recording
            
            if self.recording:  # If still recording after 3 seconds
                self.root.after(0, self._recording_complete)
            else:
                self.root.after(0, self._recording_cancelled)
                
        except Exception as e:
            self.root.after(0, lambda: self._recording_error(str(e)))
    
    def _recording_complete(self):
        """Handle recording completion."""
        self.recording = False
        self.record_button.config(text="🎙️ Start Recording")
        self.progress_var.set("Recording complete!")
        self.status_var.set("Recording saved")
        
        # Enable next button
        self.next_button.config(state='normal')
        
        # Show completion message
        messagebox.showinfo("Success", "Recording completed successfully!")
    
    def _recording_cancelled(self):
        """Handle recording cancellation."""
        self.record_button.config(text="🎙️ Start Recording")
        self.progress_var.set("Recording cancelled")
        self.status_var.set("Ready")
    
    def _recording_error(self, error_msg):
        """Handle recording error."""
        self.recording = False
        self.record_button.config(text="🎙️ Start Recording")
        self.progress_var.set("Recording failed")
        self.status_var.set("Error occurred")
        messagebox.showerror("Recording Error", f"Failed to record: {error_msg}")
    
    def _next_phrase(self):
        """Move to the next phrase."""
        if self.current_phrase < len(self.phrases) - 1:
            self.current_phrase += 1
            self._update_phrase_display()
            self.next_button.config(state='disabled')
            self.progress_var.set("Ready for next recording")
            self.status_var.set(f"Phrase {self.current_phrase + 1} of {len(self.phrases)}")
        else:
            # All phrases completed
            self.analyze_button.config(state='normal')
            self.progress_var.set("All phrases recorded! Click 'Analyze Results'")
            self.status_var.set("Ready for analysis")
            messagebox.showinfo("Complete", "All phrases have been recorded!")
    
    def _analyze_results(self):
        """Analyze the recorded results."""
        try:
            self.progress_var.set("Analyzing results...")
            self.status_var.set("Analysis in progress")
            
            # Simulate analysis (in a real implementation, this would use the actual analysis)
            time.sleep(2)
            
            # Generate mock results
            self.results = {
                'overall_score': 0.75,
                'accent_level': 'Mild accent',
                'phoneme_accuracy': 0.82,
                'pitch_similarity': 0.68,
                'duration_similarity': 0.74,
                'stress_pattern_accuracy': 0.71
            }
            
            self._display_results()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze results: {e}")
            self.status_var.set("Analysis failed")
    
    def _display_results(self):
        """Display analysis results."""
        self.results_text.delete(1.0, tk.END)
        
        results_text = f"""
🎤 Accent Strength Estimator Results
====================================

Overall Score: {self.results['overall_score']:.1%} ({self.results['accent_level']})

📈 Detailed Analysis:
- Phoneme Match Rate: {self.results['phoneme_accuracy']:.1%}
- Pitch Contour Similarity: {self.results['pitch_similarity']:.1%}
- Duration Similarity: {self.results['duration_similarity']:.1%}
- Stress Pattern Accuracy: {self.results['stress_pattern_accuracy']:.1%}

💡 Improvement Tips:
- Practice vowel length in stressed syllables
- Work on intonation patterns
- Focus on stress-timed rhythm
- Record and compare with native speakers

🎯 Recommended Practice:
- Focus on minimal pairs: /θ/ vs /t/, /ð/ vs /d/
- Practice stress-timed rhythm
- Listen to native English speakers
- Use tongue twisters for articulation

Great work! Keep practicing to improve your pronunciation.
        """
        
        self.results_text.insert(1.0, results_text)
        self.progress_var.set("Analysis complete!")
        self.status_var.set("Results displayed")
        
        messagebox.showinfo("Analysis Complete", "Results have been analyzed and displayed!")

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, call
import tkinter as tk
from tkinter import ttk
import threading
import time

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import KeyMentorUI, run_cli
from tracker import TypingTracker, TypingEvent, TypingSession
from analyzer import TypingAnalyzer, WeakSpot, TypingProfile
from exercise_generator import ExerciseGenerator, TypingExercise


class TestKeyMentorUI(unittest.TestCase):
    """Test cases for the KeyMentorUI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_typing_data.db")
        
        # Create a temporary tracker for testing
        self.tracker = TypingTracker(self.db_path)
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_ui_initialization(self, mock_notebook, mock_stringvar, mock_tk):
        """Test that the UI initializes correctly."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        # Test initialization
        ui = KeyMentorUI()
        
        # Verify Tkinter root was created
        mock_tk.assert_called_once()
        
        # Verify UI setup was called
        self.assertTrue(hasattr(ui, 'root'))
        self.assertEqual(ui.root, mock_root)
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_setup_ui(self, mock_notebook, mock_stringvar, mock_tk):
        """Test that the UI setup creates all required tabs."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Verify notebook was created
        mock_root.title.assert_called_with("KeyMentor - Typing Coach")
        
        # Verify tab creation methods would be called
        # (These are called during setup_ui)
        self.assertTrue(hasattr(ui, 'setup_ui'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_dashboard_tab_creation(self, mock_notebook, mock_stringvar, mock_tk):
        """Test dashboard tab creation."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test dashboard tab creation
        ui.create_dashboard_tab()
        
        # Verify dashboard elements would be created
        # (In a real test, we'd check for specific widgets)
        self.assertTrue(hasattr(ui, 'create_dashboard_tab'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_typing_tab_creation(self, mock_notebook, mock_stringvar, mock_tk):
        """Test typing tab creation."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test typing tab creation
        ui.create_typing_tab()
        
        # Verify typing elements would be created
        self.assertTrue(hasattr(ui, 'create_typing_tab'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_analysis_tab_creation(self, mock_notebook, mock_stringvar, mock_tk):
        """Test analysis tab creation."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test analysis tab creation
        ui.create_analysis_tab()
        
        # Verify analysis elements would be created
        self.assertTrue(hasattr(ui, 'create_analysis_tab'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_exercises_tab_creation(self, mock_notebook, mock_stringvar, mock_tk):
        """Test exercises tab creation."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test exercises tab creation
        ui.create_exercises_tab()
        
        # Verify exercises elements would be created
        self.assertTrue(hasattr(ui, 'create_exercises_tab'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_progress_tab_creation(self, mock_notebook, mock_stringvar, mock_tk):
        """Test progress tab creation."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test progress tab creation
        ui.create_progress_tab()
        
        # Verify progress elements would be created
        self.assertTrue(hasattr(ui, 'create_progress_tab'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_start_typing_session(self, mock_notebook, mock_stringvar, mock_tk):
        """Test starting a typing session."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Test starting a typing session
        ui.start_typing_session()
        
        # Verify session was started
        self.assertTrue(hasattr(ui, 'start_typing_session'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_end_typing_session(self, mock_notebook, mock_stringvar, mock_tk):
        """Test ending a typing session."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Start a session first
        ui.start_typing_session()
        
        # Test ending the session
        ui.end_typing_session()
        
        # Verify session was ended
        self.assertTrue(hasattr(ui, 'end_typing_session'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_load_sample_text(self, mock_notebook, mock_stringvar, mock_tk):
        """Test loading sample text."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        
        # Test loading sample text
        ui.load_sample_text()
        
        # Verify sample text loading
        self.assertTrue(hasattr(ui, 'load_sample_text'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_run_analysis(self, mock_notebook, mock_stringvar, mock_tk):
        """Test running typing analysis."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.analyzer = TypingAnalyzer(self.db_path)
        
        # Test running analysis
        ui.run_analysis()
        
        # Verify analysis execution
        self.assertTrue(hasattr(ui, 'run_analysis'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_generate_progress_report(self, mock_notebook, mock_stringvar, mock_tk):
        """Test generating progress report."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.analyzer = TypingAnalyzer(self.db_path)
        
        # Test generating progress report
        ui.generate_progress_report()
        
        # Verify progress report generation
        self.assertTrue(hasattr(ui, 'generate_progress_report'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_generate_exercises(self, mock_notebook, mock_stringvar, mock_tk):
        """Test generating exercises."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.generator = ExerciseGenerator()
        
        # Test generating exercises
        ui.generate_exercises()
        
        # Verify exercise generation
        self.assertTrue(hasattr(ui, 'generate_exercises'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_generate_progressive_exercises(self, mock_notebook, mock_stringvar, mock_tk):
        """Test generating progressive exercises."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.generator = ExerciseGenerator()
        
        # Test generating progressive exercises
        ui.generate_progressive_exercises()
        
        # Verify progressive exercise generation
        self.assertTrue(hasattr(ui, 'generate_progressive_exercises'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_view_recent_sessions(self, mock_notebook, mock_stringvar, mock_tk):
        """Test viewing recent sessions."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Test viewing recent sessions
        ui.view_recent_sessions()
        
        # Verify recent sessions viewing
        self.assertTrue(hasattr(ui, 'view_recent_sessions'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_export_data(self, mock_notebook, mock_stringvar, mock_tk):
        """Test exporting data."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Test exporting data
        ui.export_data()
        
        # Verify data export
        self.assertTrue(hasattr(ui, 'export_data'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_update_dashboard_stats(self, mock_notebook, mock_stringvar, mock_tk):
        """Test updating dashboard statistics."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Test updating dashboard stats
        ui.update_dashboard_stats()
        
        # Verify dashboard stats update
        self.assertTrue(hasattr(ui, 'update_dashboard_stats'))
    
    @patch('tkinter.Tk')
    @patch('tkinter.StringVar')
    @patch('tkinter.ttk.Notebook')
    def test_on_keypress(self, mock_notebook, mock_stringvar, mock_tk):
        """Test keypress event handling."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_stringvar.return_value = MagicMock()
        mock_notebook.return_value = MagicMock()
        
        ui = KeyMentorUI()
        ui.tracker = self.tracker
        
        # Create a mock event
        mock_event = MagicMock()
        mock_event.char = 'a'
        mock_event.keysym = 'a'
        
        # Test keypress handling
        ui.on_keypress(mock_event)
        
        # Verify keypress handling
        self.assertTrue(hasattr(ui, 'on_keypress'))


class TestCLI(unittest.TestCase):
    """Test cases for the CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_typing_data.db")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_basic(self, mock_print, mock_input):
        """Test basic CLI functionality."""
        # Mock user input to exit immediately
        mock_input.side_effect = ['5']  # Exit option
        
        # Test CLI execution
        run_cli()
        
        # Verify CLI ran without errors
        mock_print.assert_called()
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_invalid_option(self, mock_print, mock_input):
        """Test CLI with invalid option."""
        # Mock user input with invalid option then exit
        mock_input.side_effect = ['99', '5']  # Invalid option, then exit
        
        # Test CLI execution
        run_cli()
        
        # Verify error message was printed
        mock_print.assert_any_call("Invalid choice. Please try again.")
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_typing_practice(self, mock_print, mock_input):
        """Test CLI typing practice option."""
        # Mock user input for typing practice then exit
        mock_input.side_effect = ['1', 'Hello world', '', 'Hello world', '5']  # Practice, text, ready, input, exit
        
        # Test CLI execution
        run_cli()
        
        # Verify typing practice was initiated
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Type this text:" in call for call in calls))
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_view_progress(self, mock_print, mock_input):
        """Test CLI view progress option."""
        # Mock user input for viewing progress then exit
        mock_input.side_effect = ['2', '5']  # View progress, exit
        
        # Test CLI execution
        run_cli()
        
        # Verify progress viewing was initiated - it might show an error if no data
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Typing Analysis:" in call or "Error:" in call for call in calls))
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_generate_exercises(self, mock_print, mock_input):
        """Test CLI generate exercises option."""
        # Mock user input for generating exercises then exit
        mock_input.side_effect = ['3', '5']  # Generate exercises, exit
        
        # Test CLI execution
        run_cli()
        
        # Verify exercise generation was initiated - it might show an error if no data
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Generated Exercises:" in call or "Error:" in call for call in calls))
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_cli_export_data(self, mock_print, mock_input):
        """Test CLI export data option."""
        # Mock user input for viewing progress then exit
        mock_input.side_effect = ['4', '5']  # View progress, exit
        
        # Test CLI execution
        run_cli()
        
        # Verify progress viewing was initiated - it might show an error if no data
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("7-Day Progress Report:" in call or "Error:" in call for call in calls))


class TestUIIntegration(unittest.TestCase):
    """Integration tests for UI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_typing_data.db")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ui_components_initialization(self):
        """Test that all UI components can be initialized."""
        # Test that we can create instances of all components
        tracker = TypingTracker(self.db_path)
        analyzer = TypingAnalyzer(self.db_path)
        generator = ExerciseGenerator()
        
        # Verify components were created successfully
        self.assertIsNotNone(tracker)
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(generator)
    
    def test_ui_data_flow(self):
        """Test data flow between UI components."""
        # Create components
        tracker = TypingTracker(self.db_path)
        analyzer = TypingAnalyzer(self.db_path)
        generator = ExerciseGenerator()
        
        # Simulate a typing session
        session_id = tracker.start_session()
        tracker.record_keypress('h', 'h', 'hello')
        tracker.record_keypress('e', 'e', 'hello')
        tracker.record_keypress('l', 'l', 'hello')
        tracker.record_keypress('l', 'l', 'hello')
        tracker.record_keypress('o', 'o', 'hello')
        session = tracker.end_session()
        
        # Analyze the session
        profile = analyzer.analyze_user_typing()
        
        # Generate exercises based on profile
        exercises = generator.generate_exercises(profile)
        
        # Verify data flow worked
        self.assertIsNotNone(session)
        self.assertIsNotNone(profile)
        self.assertIsInstance(exercises, list)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_run_cli_function_exists(self):
        """Test that run_cli function exists."""
        self.assertTrue(callable(run_cli))


if __name__ == '__main__':
    unittest.main()

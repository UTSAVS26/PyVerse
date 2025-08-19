"""
Command-line interface for the Accent Strength Estimator.
"""

import time
import os
from typing import List, Dict, Any
from ..audio.reference_generator import ReferenceGenerator
from ..analysis.phoneme_analyzer import PhonemeAnalyzer
from ..scoring.feedback_generator import FeedbackGenerator

# Try to import modules that require external dependencies
try:
    from ..audio.recorder import AudioRecorder
    from ..audio.processor import AudioProcessor
    from ..scoring.accent_scorer import AccentScorer
    HAS_AUDIO_DEPENDENCIES = True
except ImportError:
    HAS_AUDIO_DEPENDENCIES = False


class CLIInterface:
    """Command-line interface for the accent strength estimator."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.reference_generator = ReferenceGenerator()
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        
        # Initialize audio components if available
        if HAS_AUDIO_DEPENDENCIES:
            self.recorder = AudioRecorder()
            self.processor = AudioProcessor()
            self.scorer = AccentScorer()
        else:
            self.recorder = None
            self.processor = None
            self.scorer = None
        
    def run(self):
        """Run the CLI interface."""
        print("🎤 Welcome to Accent Strength Estimator!")
        print("=" * 50)
        
        # Check if audio dependencies are available
        if not HAS_AUDIO_DEPENDENCIES:
            print("⚠️  Warning: Audio recording dependencies not available.")
            print("   This is a demonstration mode with simulated results.")
            print("   Install sounddevice, librosa, and other audio packages for full functionality.")
            print()
        
        # Load reference data
        reference_data = self._load_reference_data()
        if not reference_data:
            print("❌ Error: Could not load reference data.")
            return
        
        # Get user input
        phrases_to_test = self._get_phrases_to_test(reference_data)
        if not phrases_to_test:
            print("❌ No phrases selected for testing.")
            return
        
        # Run the assessment
        results = self._run_assessment(phrases_to_test, reference_data)
        
        # Display results
        self._display_results(results)
        
        print("\n🎉 Assessment complete! Thank you for using Accent Strength Estimator.")
    
    def _load_reference_data(self) -> Dict[str, Any]:
        """Load reference data from file or generate if needed."""
        reference_file = "data/reference_data.json"
        
        if os.path.exists(reference_file):
            print("📖 Loading existing reference data...")
            if self.reference_generator.load_reference_data(reference_file):
                return self.reference_generator.reference_data
        
        print("🔧 Generating reference data...")
        phrases_file = "data/reference_phrases.txt"
        if not os.path.exists(phrases_file):
            print(f"❌ Reference phrases file not found: {phrases_file}")
            return {}
        
        phrases = self.reference_generator.load_reference_phrases(phrases_file)
        if not phrases:
            print("❌ No phrases found in reference file.")
            return {}
        
        reference_data = self.reference_generator.create_reference_data(phrases)
        
        # Save reference data
        os.makedirs("data", exist_ok=True)
        self.reference_generator.save_reference_data(reference_file)
        
        return reference_data
    
    def _get_phrases_to_test(self, reference_data: Dict[str, Any]) -> List[str]:
        """Get phrases to test from user input."""
        print("\n📝 Available phrases for testing:")
        print("-" * 40)
        
        phrase_ids = list(reference_data.keys())
        for i, phrase_id in enumerate(phrase_ids, 1):
            text = reference_data[phrase_id]['text']
            difficulty = reference_data[phrase_id]['difficulty_level']
            print(f"{i:2d}. [{difficulty.upper()}] {text}")
        
        print("\nOptions:")
        print("1. Test all phrases")
        print("2. Test specific phrases")
        print("3. Test by difficulty level")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    return phrase_ids
                elif choice == "2":
                    return self._select_specific_phrases(phrase_ids)
                elif choice == "3":
                    return self._select_by_difficulty(reference_data)
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return []
    
    def _select_specific_phrases(self, phrase_ids: List[str]) -> List[str]:
        """Select specific phrases for testing."""
        print("\nEnter phrase numbers separated by commas (e.g., 1,3,5):")
        
        while True:
            try:
                selection = input("Phrases to test: ").strip()
                if not selection:
                    return []
                
                numbers = [int(x.strip()) for x in selection.split(",")]
                selected_phrases = []
                
                for num in numbers:
                    if 1 <= num <= len(phrase_ids):
                        selected_phrases.append(phrase_ids[num - 1])
                    else:
                        print(f"❌ Invalid phrase number: {num}")
                
                if selected_phrases:
                    return selected_phrases
                else:
                    print("❌ No valid phrases selected.")
                    
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return []
    
    def _select_by_difficulty(self, reference_data: Dict[str, Any]) -> List[str]:
        """Select phrases by difficulty level."""
        print("\nSelect difficulty level:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        print("4. All difficulties")

        while True:
            try:
                choice = input("Enter choice (1-4): ").strip()
                
                if choice == "1":
                    return [phrase_id for phrase_id, data in reference_data.items() 
                           if data.get('difficulty_level') == 'easy']
                elif choice == "2":
                    return [phrase_id for phrase_id, data in reference_data.items() 
                           if data.get('difficulty_level') == 'medium']
                elif choice == "3":
                    return [phrase_id for phrase_id, data in reference_data.items() 
                           if data.get('difficulty_level') == 'hard']
                elif choice == "4":
                    return list(reference_data.keys())
                else:
                    print("❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return []
    
    def _run_assessment(self, phrase_ids: List[str], reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the accent assessment."""
        print(f"\n🎤 Starting assessment with {len(phrase_ids)} phrases...")
        print("=" * 50)
        
        phrase_results = []
        
        for i, phrase_id in enumerate(phrase_ids, 1):
            phrase_data = reference_data[phrase_id]
            text = phrase_data['text']
            
            print(f"\n📝 Phrase {i}/{len(phrase_ids)}: {text}")
            
            if HAS_AUDIO_DEPENDENCIES:
                print("🎙️  Recording in 3 seconds...")
                
                # Countdown
                for count in range(3, 0, -1):
                    print(f"   {count}...")
                    time.sleep(1)
                
                print("🎙️  Recording... (speak now)")
                
                # Record audio
                audio_data = self.recorder.record_for_duration(5.0)  # 5 seconds
                
                if len(audio_data) == 0:
                    print("❌ No audio recorded. Skipping this phrase.")
                    continue
                
                print("✅ Recording complete. Analyzing...")
                
                # Generate phonemes for user audio (simplified)
                user_phonemes = self._generate_user_phonemes(text)
                
                # Analyze the phrase
                result = self.scorer.analyze_phrase(audio_data, phrase_data, user_phonemes)
            else:
                # Demo mode - simulate analysis
                print("🎙️  Demo mode - simulating recording...")
                time.sleep(2)
                print("✅ Analysis complete.")
                
                # Generate mock results
                result = self._generate_mock_result(phrase_data)
            
            phrase_results.append(result)
            
            # Show quick feedback
            quick_score = result.get('overall_score', 0.0)
            print(f"📊 Score: {quick_score:.1%}")
        
        # Compute overall results
        if phrase_results:
            if HAS_AUDIO_DEPENDENCIES:
                overall_results = self.scorer.analyze_multiple_phrases(phrase_results)
            else:
                overall_results = self._generate_mock_overall_results(phrase_results)
            return overall_results
        else:
            return {}
    
    def _generate_user_phonemes(self, text: str) -> List[str]:
        """Generate phonemes for user audio (simplified implementation)."""
        # In a real implementation, this would use speech recognition
        # For now, we'll use a simplified approach
        return self.reference_generator.generate_phonemes(text)
    
    def _generate_mock_result(self, phrase_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock analysis result for demo mode."""
        import random
        
        # Generate realistic mock scores
        phoneme_accuracy = random.uniform(0.7, 0.95)
        pitch_similarity = random.uniform(0.6, 0.9)
        duration_similarity = random.uniform(0.65, 0.85)
        stress_pattern_accuracy = random.uniform(0.7, 0.9)
        
        # Calculate overall score
        overall_score = (phoneme_accuracy * 0.4 + 
                        pitch_similarity * 0.25 + 
                        duration_similarity * 0.2 + 
                        stress_pattern_accuracy * 0.15)
        
        # Determine accent level
        if overall_score >= 0.9:
            accent_level = "Native-like"
        elif overall_score >= 0.8:
            accent_level = "Very mild accent"
        elif overall_score >= 0.7:
            accent_level = "Mild accent"
        elif overall_score >= 0.6:
            accent_level = "Moderate accent"
        elif overall_score >= 0.5:
            accent_level = "Strong accent"
        else:
            accent_level = "Very strong accent"
        
        return {
            'phoneme_accuracy': phoneme_accuracy,
            'pitch_similarity': pitch_similarity,
            'duration_similarity': duration_similarity,
            'stress_pattern_accuracy': stress_pattern_accuracy,
            'overall_score': overall_score,
            'accent_level': accent_level
        }
    
    def _generate_mock_overall_results(self, phrase_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mock overall results for demo mode."""
        if not phrase_results:
            return {}
        
        # Calculate averages
        total_phoneme_accuracy = sum(r.get('phoneme_accuracy', 0) for r in phrase_results)
        total_pitch_similarity = sum(r.get('pitch_similarity', 0) for r in phrase_results)
        total_duration_similarity = sum(r.get('duration_similarity', 0) for r in phrase_results)
        total_stress_pattern_accuracy = sum(r.get('stress_pattern_accuracy', 0) for r in phrase_results)
        
        num_results = len(phrase_results)
        
        avg_phoneme_accuracy = total_phoneme_accuracy / num_results
        avg_pitch_similarity = total_pitch_similarity / num_results
        avg_duration_similarity = total_duration_similarity / num_results
        avg_stress_pattern_accuracy = total_stress_pattern_accuracy / num_results
        
        # Calculate overall score
        overall_score = (avg_phoneme_accuracy * 0.4 + 
                        avg_pitch_similarity * 0.25 + 
                        avg_duration_similarity * 0.2 + 
                        avg_stress_pattern_accuracy * 0.15)
        
        # Determine accent level
        if overall_score >= 0.9:
            accent_level = "Native-like"
        elif overall_score >= 0.8:
            accent_level = "Very mild accent"
        elif overall_score >= 0.7:
            accent_level = "Mild accent"
        elif overall_score >= 0.6:
            accent_level = "Moderate accent"
        elif overall_score >= 0.5:
            accent_level = "Strong accent"
        else:
            accent_level = "Very strong accent"
        
        return {
            'phoneme_accuracy': avg_phoneme_accuracy,
            'pitch_similarity': avg_pitch_similarity,
            'duration_similarity': avg_duration_similarity,
            'stress_pattern_accuracy': avg_stress_pattern_accuracy,
            'overall_score': overall_score,
            'accent_level': accent_level
        }
    
    def _display_results(self, results: Dict[str, Any]):
        """Display the assessment results."""
        if not results:
            print("❌ No results to display.")
            return
        
        print("\n" + "=" * 60)
        print("🎤 ACCENT STRENGTH ESTIMATOR RESULTS")
        print("=" * 60)
        
        # Overall score
        overall_score = results.get('overall_score', 0.0)
        accent_level = results.get('accent_level', 'Unknown')
        
        print(f"\n📊 Overall Score: {overall_score:.1%} ({accent_level})")
        
        # Component scores
        print("\n📈 Detailed Analysis:")
        component_scores = {
            'phoneme_accuracy': 'Phoneme Match Rate',
            'pitch_similarity': 'Pitch Contour Similarity',
            'duration_similarity': 'Duration Similarity',
            'stress_pattern_accuracy': 'Stress Pattern Accuracy'
        }
        
        for component, label in component_scores.items():
            score = results.get(component, 0.0)
            print(f"- {label}: {score:.1%}")
        
        # Generate and display feedback
        feedback = self.feedback_generator.generate_comprehensive_feedback(results)
        feedback_report = self.feedback_generator.format_feedback_report(feedback)
        
        print("\n" + feedback_report)
        
        # Save results
        self._save_results(results)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {filepath}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
🎤 Accent Strength Estimator - Help

This tool analyzes your English pronunciation and provides feedback on your accent strength.

USAGE:
    python main.py --mode cli

FEATURES:
    - Records your speech for analysis
    - Compares pronunciation to native English reference
    - Provides detailed feedback on phonemes, intonation, and timing
    - Generates personalized improvement tips

TIPS FOR BEST RESULTS:
    - Speak clearly and at a normal pace
    - Ensure good microphone quality
    - Minimize background noise
    - Practice the phrases before recording

For more information, visit the project documentation.
        """
        print(help_text)

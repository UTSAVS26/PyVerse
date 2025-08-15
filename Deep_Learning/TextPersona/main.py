#!/usr/bin/env python3
"""
TextPersona - Personality Type Predictor from Text Prompts

A complete NLP-based tool that predicts MBTI personality types from introspective text responses.
"""

import sys
import os
import argparse
from typing import Dict, Optional

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.prompt_interface import PromptInterface
from core.classifier_zero_shot import ZeroShotClassifier
from core.classifier_rules import RuleBasedClassifier
from core.result_formatter import ResultFormatter

def run_cli_mode():
    """Run the application in command-line interface mode."""
    print("🧠 TextPersona - Personality Type Predictor")
    print("=" * 50)
    
    # Initialize components
    prompt_interface = PromptInterface()
    classifier = ZeroShotClassifier()  # Try zero-shot first
    formatter = ResultFormatter()
    
    # Collect responses
    responses = prompt_interface.run_cli_interface()
    
    if not responses:
        print("❌ No responses collected. Exiting.")
        return
    
    # Format responses for classification
    formatted_text = prompt_interface.format_responses_for_classifier(responses)
    
    print("\n" + "=" * 50)
    print("🔍 Analyzing your personality...")
    
    # Classify personality
    classification_result = classifier.classify_personality(formatted_text)
    
    # Get personality description
    personality_desc = classifier.get_personality_description(
        classification_result["mbti_type"]
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 YOUR RESULTS")
    print("=" * 50)
    
    result_text = formatter.format_personality_result(classification_result, personality_desc)
    print(result_text)
    
    # Export results
    success = formatter.export_results(classification_result, personality_desc)
    if success:
        print("\n✅ Results exported to personality_results.txt")
    
    # Save responses (optional)
    prompt_interface.save_responses(responses)
    
    print("\n🎉 Analysis complete! Thank you for using TextPersona!")

def run_streamlit_mode():
    """Run the application in Streamlit web interface mode."""
    import subprocess
    import sys
    
    # Get the path to the Streamlit app
    streamlit_app_path = os.path.join(os.path.dirname(__file__), 'ui', 'streamlit_app.py')
    
    if not os.path.exists(streamlit_app_path):
        print(f"❌ Streamlit app not found at {streamlit_app_path}")
        return
    
    print("🌐 Starting Streamlit web interface...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", streamlit_app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def test_classifiers():
    """Test the classification systems with sample data."""
    print("🧪 Testing classification systems...")
    
    # Sample responses for testing
    test_responses = {
        1: "Primarily logic",
        2: "Routine and structure", 
        3: "Alone or with close friends",
        4: "Concrete details and facts",
        5: "Achieving goals and success",
        6: "Plan carefully and stick to the plan",
        7: "Observe and listen",
        8: "Analyze thoroughly with facts",
        9: "Working alone",
        10: "Step-by-step instructions"
    }
    
    # Initialize components
    prompt_interface = PromptInterface()
    zero_shot_classifier = ZeroShotClassifier()
    rule_based_classifier = RuleBasedClassifier()
    formatter = ResultFormatter()
    
    # Format test responses
    formatted_text = prompt_interface.format_responses_for_classifier(test_responses)
    
    print(f"📝 Test text length: {len(formatted_text)} characters")
    
    # Test zero-shot classifier
    print("\n🔬 Testing Zero-shot Classifier...")
    try:
        zero_shot_result = zero_shot_classifier.classify_personality(formatted_text)
        print(f"✅ Zero-shot result: {zero_shot_result['mbti_type']} (confidence: {zero_shot_result['confidence']:.1%})")
    except Exception as e:
        print(f"❌ Zero-shot classifier failed: {e}")
    
    # Test rule-based classifier
    print("\n🔬 Testing Rule-based Classifier...")
    try:
        rule_based_result = rule_based_classifier.classify_personality(formatted_text)
        print(f"✅ Rule-based result: {rule_based_result['mbti_type']} (confidence: {rule_based_result['confidence']:.1%})")
    except Exception as e:
        print(f"❌ Rule-based classifier failed: {e}")
    
    print("\n✅ Testing complete!")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="TextPersona - Personality Type Predictor from Text Prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run CLI interface
  python main.py --web             # Run Streamlit web interface
  python main.py --test            # Run classifier tests
  python main.py --cli             # Run CLI interface (explicit)
        """
    )
    
    parser.add_argument(
        "--web", 
        action="store_true",
        help="Run the Streamlit web interface"
    )
    
    parser.add_argument(
        "--cli",
        action="store_true", 
        help="Run the command-line interface"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run classifier tests"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.test:
        test_classifiers()
    elif args.web:
        run_streamlit_mode()
    else:
        # Default to CLI mode
        run_cli_mode()

if __name__ == "__main__":
    main() 
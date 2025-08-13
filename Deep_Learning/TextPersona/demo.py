#!/usr/bin/env python3
"""
TextPersona Demo Script

This script demonstrates the complete TextPersona functionality with sample data.
"""

import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.prompt_interface import PromptInterface
from core.classifier_zero_shot import ZeroShotClassifier
from core.classifier_rules import RuleBasedClassifier
from core.result_formatter import ResultFormatter

def run_demo():
    """Run a demonstration of the TextPersona system."""
    print("🧠 TextPersona - Personality Type Predictor Demo")
    print("=" * 60)
    
    # Sample responses for demonstration
    demo_responses = {
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
    
    print("\n📝 Sample Responses:")
    for q_id, response in demo_responses.items():
        print(f"Q{q_id}: {response}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    interface = PromptInterface()
    zero_shot_classifier = ZeroShotClassifier()
    rule_based_classifier = RuleBasedClassifier()
    formatter = ResultFormatter()
    
    # Format responses
    print("\n📊 Formatting responses for analysis...")
    formatted_text = interface.format_responses_for_classifier(demo_responses)
    print(f"Formatted text length: {len(formatted_text)} characters")
    
    # Test both classifiers
    print("\n🤖 Testing Zero-shot Classifier...")
    try:
        zero_shot_result = zero_shot_classifier.classify_personality(formatted_text)
        zero_shot_desc = zero_shot_classifier.get_personality_description(
            zero_shot_result["mbti_type"]
        )
        print(f"✅ Zero-shot result: {zero_shot_result['mbti_type']} "
              f"({zero_shot_desc['name']}) - Confidence: {zero_shot_result['confidence']:.1%}")
    except Exception as e:
        print(f"❌ Zero-shot classifier failed: {e}")
    
    print("\n🔍 Testing Rule-based Classifier...")
    try:
        rule_based_result = rule_based_classifier.classify_personality(formatted_text)
        rule_based_desc = rule_based_classifier.get_personality_description(
            rule_based_result["mbti_type"]
        )
        print(f"✅ Rule-based result: {rule_based_result['mbti_type']} "
              f"({rule_based_desc['name']}) - Confidence: {rule_based_result['confidence']:.1%}")
    except Exception as e:
        print(f"❌ Rule-based classifier failed: {e}")
    
    # Show detailed results
    print("\n📋 Detailed Results:")
    print("-" * 40)
    
    # Use the rule-based result for detailed display
    if 'rule_based_result' in locals():
        result_text = formatter.format_personality_result(rule_based_result, rule_based_desc)
        print(result_text)
        
        # Export results
        print("\n💾 Exporting results...")
        success = formatter.export_results(rule_based_result, rule_based_desc, "demo_results.txt")
        if success:
            print("✅ Results exported to demo_results.txt")
        else:
            print("❌ Failed to export results")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full application:")
    print("  python main.py          # CLI interface")
    print("  python main.py --web    # Web interface")
    print("  python main.py --test   # Run tests")

if __name__ == "__main__":
    run_demo() 
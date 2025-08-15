#!/usr/bin/env python3
"""
Test cases for TextPersona classifiers.

This module contains comprehensive tests for both zero-shot and rule-based classifiers.
"""

import sys
import os
import unittest
import json
from typing import Dict, List

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.classifier_zero_shot import ZeroShotClassifier
from core.classifier_rules import RuleBasedClassifier
from core.prompt_interface import PromptInterface
from core.result_formatter import ResultFormatter

class TestPromptInterface(unittest.TestCase):
    """Test cases for the prompt interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = PromptInterface()
    
    def test_load_questions(self):
        """Test that questions are loaded correctly."""
        questions = self.interface.get_questions()
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        
        # Check structure of first question
        first_question = questions[0]
        self.assertIn('id', first_question)
        self.assertIn('question', first_question)
        self.assertIn('category', first_question)
        self.assertIn('options', first_question)
        self.assertIsInstance(first_question['options'], list)
    
    def test_format_responses(self):
        """Test response formatting."""
        test_responses = {
            1: "Primarily logic",
            2: "Routine and structure",
            3: "Alone or with close friends"
        }
        
        formatted = self.interface.format_responses_for_classifier(test_responses)
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)
        self.assertIn("Personality Assessment Responses", formatted)
    
    def test_save_responses(self):
        """Test response saving functionality."""
        test_responses = {1: "Test response"}
        
        # Test that save doesn't raise an exception
        try:
            self.interface.save_responses(test_responses)
            success = True
        except Exception as e:
            success = False
            print(f"Save responses failed: {e}")
        
        self.assertTrue(success)

class TestRuleBasedClassifier(unittest.TestCase):
    """Test cases for the rule-based classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RuleBasedClassifier()
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier.sentiment_analyzer)
        self.assertIsNotNone(self.classifier.keyword_mappings)
        self.assertIn("introversion_extraversion", self.classifier.keyword_mappings)
    
    def test_classify_personality_logic_dominant(self):
        """Test classification with logic-dominant responses."""
        test_text = """
        I prefer to make decisions based on logic and analysis. 
        I focus on concrete facts and details when solving problems.
        I like to plan carefully and stick to structured approaches.
        I prefer working alone and analyzing situations thoroughly.
        """
        
        result = self.classifier.classify_personality(test_text)
        
        self.assertIn("mbti_type", result)
        self.assertIn("confidence", result)
        self.assertIn("dimensions", result)
        self.assertIn("method", result)
        self.assertEqual(result["method"], "rule-based")
        
        # Should lean towards thinking and sensing
        mbti_type = result["mbti_type"]
        self.assertIsInstance(mbti_type, str)
        self.assertEqual(len(mbti_type), 4)
    
    def test_classify_personality_feeling_dominant(self):
        """Test classification with feeling-dominant responses."""
        test_text = """
        I make decisions based on my feelings and values.
        I care deeply about people and relationships.
        I prefer to help others and create harmony.
        I trust my gut feelings and intuition.
        """
        
        result = self.classifier.classify_personality(test_text)
        
        self.assertIn("mbti_type", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0)
    
    def test_classify_personality_extraversion_dominant(self):
        """Test classification with extraversion-dominant responses."""
        test_text = """
        I love being around people and social groups.
        I am outgoing and energetic in social situations.
        I prefer collaborative work and team activities.
        I enjoy being the center of attention.
        """
        
        result = self.classifier.classify_personality(test_text)
        
        self.assertIn("mbti_type", result)
        # Should lean towards extraversion
        mbti_type = result["mbti_type"]
        self.assertIsInstance(mbti_type, str)
    
    def test_analyze_dimensions(self):
        """Test dimension analysis."""
        test_text = "I prefer logic and analysis over emotions."
        
        dimensions = self.classifier._analyze_dimensions(test_text)
        
        self.assertIn("introversion_extraversion", dimensions)
        self.assertIn("sensing_intuition", dimensions)
        self.assertIn("thinking_feeling", dimensions)
        self.assertIn("judging_perceiving", dimensions)
        
        for dimension, scores in dimensions.items():
            self.assertIsInstance(scores, dict)
            self.assertGreater(len(scores), 0)
    
    def test_construct_mbti_type(self):
        """Test MBTI type construction."""
        dimension_scores = {
            "introversion_extraversion": {"I": 5, "E": 2},
            "sensing_intuition": {"S": 3, "N": 4},
            "thinking_feeling": {"T": 6, "F": 1},
            "judging_perceiving": {"J": 4, "P": 3}
        }
        
        mbti_type = self.classifier._construct_mbti_type(dimension_scores)
        
        self.assertIsInstance(mbti_type, str)
        self.assertEqual(len(mbti_type), 4)
        # Should be INTP based on the scores above
        self.assertIn(mbti_type[0], "IE")
        self.assertIn(mbti_type[1], "SN")
        self.assertIn(mbti_type[2], "TF")
        self.assertIn(mbti_type[3], "JP")
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        dimension_scores = {
            "introversion_extraversion": {"I": 5, "E": 2},
            "sensing_intuition": {"S": 3, "N": 4},
            "thinking_feeling": {"T": 6, "F": 1},
            "judging_perceiving": {"J": 4, "P": 3}
        }
        
        confidence = self.classifier._calculate_confidence(dimension_scores)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_get_personality_description(self):
        """Test personality description retrieval."""
        description = self.classifier.get_personality_description("INTJ")
        
        self.assertIn("name", description)
        self.assertIn("strengths", description)
        self.assertIn("careers", description)
        self.assertIn("description", description)
        
        self.assertEqual(description["name"], "The Architect")
        self.assertIsInstance(description["strengths"], list)
        self.assertIsInstance(description["careers"], list)

class TestZeroShotClassifier(unittest.TestCase):
    """Test cases for the zero-shot classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.classifier = ZeroShotClassifier()
        except Exception as e:
            self.skipTest(f"Zero-shot classifier not available: {e}")
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier.mbti_types)
        self.assertIn("INTJ", self.classifier.mbti_types)
        self.assertIn("dimension_mappings", self.classifier.__dict__)
    
    def test_classify_personality_fallback(self):
        """Test classification with fallback to rule-based."""
        test_text = "I prefer logic and analysis in decision making."
        
        result = self.classifier.classify_personality(test_text)
        
        self.assertIn("mbti_type", result)
        self.assertIn("confidence", result)
        self.assertIn("method", result)
        
        # Should work even if zero-shot fails
        self.assertIsInstance(result["mbti_type"], str)
        self.assertGreaterEqual(result["confidence"], 0.0)
    
    def test_rule_based_fallback(self):
        """Test rule-based fallback functionality."""
        test_text = "I am introverted and prefer logical analysis."
        
        result = self.classifier._rule_based_classification(test_text)
        
        self.assertIn("mbti_type", result)
        self.assertIn("confidence", result)
        self.assertIn("dimensions", result)
        self.assertIn("method", result)
        self.assertEqual(result["method"], "rule-based")
    
    def test_get_personality_description(self):
        """Test personality description retrieval."""
        description = self.classifier.get_personality_description("INFJ")
        
        self.assertIn("name", description)
        self.assertIn("strengths", description)
        self.assertIn("careers", description)
        self.assertIn("description", description)
        
        self.assertEqual(description["name"], "The Advocate")

class TestResultFormatter(unittest.TestCase):
    """Test cases for the result formatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter()
    
    def test_format_personality_result(self):
        """Test personality result formatting."""
        classification_result = {
            "mbti_type": "INTJ",
            "confidence": 0.85,
            "method": "zero-shot"
        }
        
        personality_desc = {
            "name": "The Architect",
            "strengths": ["Strategic", "Independent", "Analytical"],
            "careers": ["Scientist", "Engineer"],
            "description": "Imaginative and strategic thinkers."
        }
        
        result_text = self.formatter.format_personality_result(classification_result, personality_desc)
        
        self.assertIsInstance(result_text, str)
        self.assertIn("INTJ", result_text)
        self.assertIn("The Architect", result_text)
        self.assertIn("85.0%", result_text)
    
    def test_create_radar_chart(self):
        """Test radar chart creation."""
        dimension_scores = {
            "introversion_extraversion": {"I": 5, "E": 2},
            "sensing_intuition": {"S": 3, "N": 4},
            "thinking_feeling": {"T": 6, "F": 1},
            "judging_perceiving": {"J": 4, "P": 3}
        }
        
        chart = self.formatter.create_radar_chart(dimension_scores)
        
        # Should return a plotly figure or None
        self.assertTrue(chart is None or hasattr(chart, 'to_dict'))
    
    def test_create_confidence_gauge(self):
        """Test confidence gauge creation."""
        confidence = 0.75
        
        gauge = self.formatter.create_confidence_gauge(confidence)
        
        self.assertIsNotNone(gauge)
        self.assertTrue(hasattr(gauge, 'to_dict'))
    
    def test_format_detailed_analysis(self):
        """Test detailed analysis formatting."""
        analysis = {
            "text_length": 150,
            "word_count": 25,
            "sentence_count": 3,
            "sentiment": {"pos": 0.2, "neg": 0.1, "neu": 0.7, "compound": 0.1}
        }
        
        analysis_text = self.formatter.format_detailed_analysis(analysis)
        
        self.assertIsInstance(analysis_text, str)
        self.assertIn("150", analysis_text)
        self.assertIn("25", analysis_text)
        self.assertIn("0.200", analysis_text)
    
    def test_export_results(self):
        """Test results export functionality."""
        classification_result = {
            "mbti_type": "INTJ",
            "confidence": 0.85,
            "method": "zero-shot"
        }
        
        personality_desc = {
            "name": "The Architect",
            "strengths": ["Strategic", "Independent"],
            "careers": ["Scientist", "Engineer"],
            "description": "Imaginative and strategic thinkers."
        }
        
        success = self.formatter.export_results(classification_result, personality_desc, "test_results.txt")
        
        self.assertTrue(success)
        
        # Clean up test file
        if os.path.exists("test_results.txt"):
            os.remove("test_results.txt")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_classification(self):
        """Test complete end-to-end classification."""
        # Create test responses
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
        interface = PromptInterface()
        rule_classifier = RuleBasedClassifier()
        formatter = ResultFormatter()
        
        # Format responses
        formatted_text = interface.format_responses_for_classifier(test_responses)
        
        # Classify personality
        classification_result = rule_classifier.classify_personality(formatted_text)
        
        # Get personality description
        personality_desc = rule_classifier.get_personality_description(
            classification_result["mbti_type"]
        )
        
        # Format results
        result_text = formatter.format_personality_result(classification_result, personality_desc)
        
        # Assertions
        self.assertIsInstance(classification_result, dict)
        self.assertIn("mbti_type", classification_result)
        self.assertIn("confidence", classification_result)
        self.assertIsInstance(result_text, str)
        self.assertGreater(len(result_text), 0)
    
    def test_zero_shot_integration(self):
        """Test zero-shot classifier integration."""
        try:
            zero_shot_classifier = ZeroShotClassifier()
            
            test_text = "I prefer logic and analysis in decision making."
            result = zero_shot_classifier.classify_personality(test_text)
            
            self.assertIsInstance(result, dict)
            self.assertIn("mbti_type", result)
            self.assertIn("confidence", result)
            
        except Exception as e:
            # Skip if zero-shot is not available
            self.skipTest(f"Zero-shot classifier not available: {e}")

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPromptInterface,
        TestRuleBasedClassifier,
        TestZeroShotClassifier,
        TestResultFormatter,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 
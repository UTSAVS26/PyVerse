import os
import json
from typing import Dict, List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class ZeroShotClassifier:
    """Zero-shot classification for personality type prediction using pre-trained models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize the zero-shot classifier."""
        self.model_name = model_name
        self.classifier = None
        self.mbti_types = [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP"
        ]
        
        # MBTI dimension mappings
        self.dimension_mappings = {
            "introversion_extraversion": {
                "I": ["introverted", "introversion", "alone", "solitude", "internal", "quiet"],
                "E": ["extraverted", "extraversion", "social", "group", "external", "outgoing"]
            },
            "sensing_intuition": {
                "S": ["sensing", "concrete", "facts", "details", "practical", "experience"],
                "N": ["intuition", "abstract", "concepts", "possibilities", "imaginative", "theoretical"]
            },
            "thinking_feeling": {
                "T": ["thinking", "logic", "analysis", "objective", "fairness", "tasks"],
                "F": ["feeling", "emotion", "values", "harmony", "subjective", "people"]
            },
            "judging_perceiving": {
                "J": ["judging", "structure", "planning", "organized", "closure", "decisions"],
                "P": ["perceiving", "flexibility", "spontaneity", "adaptable", "options", "openness"]
            }
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the zero-shot classification model."""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Loaded zero-shot classifier: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to rule-based classification...")
            self.classifier = None
    
    def classify_personality(self, text: str) -> Dict:
        """Classify personality type from text responses."""
        if self.classifier is None:
            return self._rule_based_classification(text)
        
        try:
            # First, classify by MBTI dimensions
            dimension_results = self._classify_dimensions(text)
            
            # Then, classify by full MBTI types
            mbti_result = self._classify_mbti_types(text)
            
            return {
                "mbti_type": mbti_result["predicted_type"],
                "confidence": mbti_result["confidence"],
                "dimensions": dimension_results,
                "method": "zero-shot"
            }
            
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return self._rule_based_classification(text)
    
    def _classify_dimensions(self, text: str) -> Dict:
        """Classify each MBTI dimension separately."""
        dimension_results = {}
        
        for dimension, mappings in self.dimension_mappings.items():
            candidate_labels = []
            for letter, keywords in mappings.items():
                candidate_labels.append(f"{letter} ({', '.join(keywords[:3])})")
            
            try:
                result = self.classifier(
                    text,
                    candidate_labels,
                    hypothesis_template="This text describes someone who is {}."
                )
                
                # Extract the predicted letter (I/E, S/N, T/F, J/P)
                predicted_label = result["labels"][0]
                predicted_letter = predicted_label.split(" ")[0]
                confidence = result["scores"][0]
                
                dimension_results[dimension] = {
                    "predicted": predicted_letter,
                    "confidence": confidence,
                    "all_scores": dict(zip(result["labels"], result["scores"]))
                }
                
            except Exception as e:
                print(f"Error classifying dimension {dimension}: {e}")
                dimension_results[dimension] = {
                    "predicted": "U",  # Unknown
                    "confidence": 0.0,
                    "all_scores": {}
                }
        
        return dimension_results
    
    def _classify_mbti_types(self, text: str) -> Dict:
        """Classify the full MBTI type."""
        try:
            result = self.classifier(
                text,
                self.mbti_types,
                hypothesis_template="This text describes someone with personality type {}."
            )
            
            return {
                "predicted_type": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
            
        except Exception as e:
            print(f"Error in MBTI type classification: {e}")
            return {
                "predicted_type": "UNKNOWN",
                "confidence": 0.0,
                "all_scores": {}
            }
    
    def _rule_based_classification(self, text: str) -> Dict:
        """Fallback rule-based classification using keyword matching."""
        text_lower = text.lower()
        
        # Initialize dimension scores
        dimension_scores = {
            "introversion_extraversion": {"I": 0, "E": 0},
            "sensing_intuition": {"S": 0, "N": 0},
            "thinking_feeling": {"T": 0, "F": 0},
            "judging_perceiving": {"J": 0, "P": 0}
        }
        
        # Score each dimension based on keyword presence
        for dimension, mappings in self.dimension_mappings.items():
            for letter, keywords in mappings.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        dimension_scores[dimension][letter] += 1
        
        # Determine predicted dimensions
        predicted_dimensions = {}
        for dimension, scores in dimension_scores.items():
            if dimension == "introversion_extraversion":
                if scores["I"] > scores["E"]:
                    predicted_dimensions[dimension] = "I"
                elif scores["E"] > scores["I"]:
                    predicted_dimensions[dimension] = "E"
                else:
                    predicted_dimensions[dimension] = "U"  # Undecided
            elif dimension == "sensing_intuition":
                if scores["S"] > scores["N"]:
                    predicted_dimensions[dimension] = "S"
                elif scores["N"] > scores["S"]:
                    predicted_dimensions[dimension] = "N"
                else:
                    predicted_dimensions[dimension] = "U"
            elif dimension == "thinking_feeling":
                if scores["T"] > scores["F"]:
                    predicted_dimensions[dimension] = "T"
                elif scores["F"] > scores["T"]:
                    predicted_dimensions[dimension] = "F"
                else:
                    predicted_dimensions[dimension] = "U"
            elif dimension == "judging_perceiving":
                if scores["J"] > scores["P"]:
                    predicted_dimensions[dimension] = "J"
                elif scores["P"] > scores["J"]:
                    predicted_dimensions[dimension] = "P"
                else:
                    predicted_dimensions[dimension] = "U"
        
        # Construct MBTI type
        mbti_type = ""
        for dimension in ["introversion_extraversion", "sensing_intuition", "thinking_feeling", "judging_perceiving"]:
            pred = predicted_dimensions.get(dimension, "U")
            if pred != "U":
                mbti_type += pred
        
        # If we couldn't determine all dimensions, use a default
        if len(mbti_type) < 4:
            mbti_type = "INTJ"  # Default fallback
        
        return {
            "mbti_type": mbti_type,
            "confidence": 0.5,  # Lower confidence for rule-based
            "dimensions": predicted_dimensions,
            "method": "rule-based"
        }
    
    def get_personality_description(self, mbti_type: str) -> Dict:
        """Get personality description for a given MBTI type."""
        descriptions = {
            "INTJ": {
                "name": "The Architect",
                "strengths": ["Strategic", "Independent", "Analytical", "Determined"],
                "careers": ["Scientist", "Engineer", "Strategic Planner", "Investment Banker"],
                "description": "Imaginative and strategic thinkers with a plan for everything."
            },
            "INTP": {
                "name": "The Logician",
                "strengths": ["Innovative", "Objective", "Logical", "Abstract"],
                "careers": ["Software Developer", "Researcher", "Philosopher", "Data Analyst"],
                "description": "Innovative inventors with an unquenchable thirst for knowledge."
            },
            "INFJ": {
                "name": "The Advocate",
                "strengths": ["Creative", "Insightful", "Inspiring", "Decisive"],
                "careers": ["Counselor", "Writer", "Psychologist", "Teacher"],
                "description": "Quiet and mystical, yet very inspiring and tireless idealists."
            },
            "INFP": {
                "name": "The Mediator",
                "strengths": ["Idealistic", "Seeks Harmony", "Open-minded", "Creative"],
                "careers": ["Artist", "Writer", "Counselor", "Social Worker"],
                "description": "Poetic, kind and altruistic people, always eager to help a good cause."
            },
            "ENFJ": {
                "name": "The Protagonist",
                "strengths": ["Receptive", "Reliable", "Charismatic", "Altruistic"],
                "careers": ["Teacher", "HR Manager", "Sales Manager", "Non-profit Director"],
                "description": "Charismatic and inspiring leaders, able to mesmerize their listeners."
            },
            "ENFP": {
                "name": "The Campaigner",
                "strengths": ["Curious", "Enthusiastic", "Excellent Communicators", "Know How to Relax"],
                "careers": ["Journalist", "Actor", "Sales Representative", "Event Planner"],
                "description": "Enthusiastic, creative and sociable free spirits."
            },
            "ENTJ": {
                "name": "The Commander",
                "strengths": ["Bold", "Imaginative", "Strong-willed", "Natural Leader"],
                "careers": ["CEO", "Entrepreneur", "Lawyer", "Military Officer"],
                "description": "Bold, imaginative and strong-willed leaders."
            },
            "ENTP": {
                "name": "The Debater",
                "strengths": ["Knowledgeable", "Quick-thinking", "Excellent Brainstormers"],
                "careers": ["Entrepreneur", "Consultant", "Lawyer", "Marketing Manager"],
                "description": "Smart and curious thinkers who cannot resist an intellectual challenge."
            }
        }
        
        # Add more types as needed...
        descriptions.update({
            "ISTJ": {
                "name": "The Logistician",
                "strengths": ["Practical", "Fact-minded", "Reliable", "Direct"],
                "careers": ["Accountant", "Military Officer", "Manager", "Auditor"],
                "description": "Practical and fact-minded individuals, whose reliability cannot be doubted."
            },
            "ISFJ": {
                "name": "The Defender",
                "strengths": ["Supportive", "Reliable", "Patient", "Imaginative"],
                "careers": ["Nurse", "Teacher", "Social Worker", "Administrative Assistant"],
                "description": "Very dedicated and warm protectors, always ready to defend their loved ones."
            },
            "ESTJ": {
                "name": "The Executive",
                "strengths": ["Dedicated", "Strong-willed", "Direct", "Loyal"],
                "careers": ["Military Officer", "Manager", "Financial Advisor", "Real Estate Agent"],
                "description": "Excellent administrators, unsurpassed at managing things or people."
            },
            "ESFJ": {
                "name": "The Consul",
                "strengths": ["Extraordinarily Caring", "Social", "Popular", "Reliable"],
                "careers": ["Nurse", "Teacher", "Sales Representative", "Customer Service"],
                "description": "Extraordinarily caring, social and popular people."
            },
            "ISTP": {
                "name": "The Virtuoso",
                "strengths": ["Optimistic", "Energetic", "Creative", "Practical"],
                "careers": ["Mechanic", "Engineer", "Pilot", "Athlete"],
                "description": "Bold and practical experimenters, masters of all kinds of tools."
            },
            "ISFP": {
                "name": "The Adventurer",
                "strengths": ["Charming", "Sensitive to Others", "Imaginative", "Artistic"],
                "careers": ["Artist", "Designer", "Photographer", "Veterinarian"],
                "description": "Flexible and charming artists, always ready to explore and experience something new."
            },
            "ESTP": {
                "name": "The Entrepreneur",
                "strengths": ["Bold", "Rational", "Practical", "Original"],
                "careers": ["Entrepreneur", "Sales Representative", "Police Officer", "Firefighter"],
                "description": "Smart, energetic and very perceptive people."
            },
            "ESFP": {
                "name": "The Entertainer",
                "strengths": ["Bold", "Original", "Practical", "Excellent People Skills"],
                "careers": ["Actor", "Event Planner", "Sales Representative", "Teacher"],
                "description": "Spontaneous, energetic and enthusiastic entertainers."
            }
        })
        
        return descriptions.get(mbti_type, {
            "name": "Unknown Type",
            "strengths": ["Unknown"],
            "careers": ["Various"],
            "description": "Personality type not fully determined."
        }) 
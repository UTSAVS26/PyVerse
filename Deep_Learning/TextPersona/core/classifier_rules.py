import re
import nltk
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

class RuleBasedClassifier:
    """Rule-based personality classification using NLP heuristics and keyword analysis."""
    
    def __init__(self):
        """Initialize the rule-based classifier."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Keyword mappings for MBTI dimensions
        self.keyword_mappings = {
            "introversion_extraversion": {
                "I": [
                    "alone", "solitude", "quiet", "internal", "reflective", "contemplative",
                    "private", "reserved", "thoughtful", "introspective", "independent",
                    "self-contained", "reserved", "shy", "withdrawn", "meditative"
                ],
                "E": [
                    "social", "outgoing", "extroverted", "energetic", "talkative", "expressive",
                    "group", "team", "collaborative", "interactive", "engaging", "enthusiastic",
                    "people-oriented", "sociable", "gregarious", "lively", "dynamic"
                ]
            },
            "sensing_intuition": {
                "S": [
                    "concrete", "facts", "details", "practical", "experience", "realistic",
                    "specific", "tangible", "observable", "measurable", "hands-on",
                    "step-by-step", "methodical", "systematic", "organized", "structured"
                ],
                "N": [
                    "abstract", "concepts", "possibilities", "imaginative", "theoretical",
                    "creative", "innovative", "visionary", "intuitive", "inspiration",
                    "big-picture", "holistic", "metaphorical", "symbolic", "artistic"
                ]
            },
            "thinking_feeling": {
                "T": [
                    "logic", "analysis", "objective", "fairness", "tasks", "efficiency",
                    "rational", "systematic", "analytical", "critical", "detached",
                    "impersonal", "fact-based", "evidence", "reasoning", "problem-solving"
                ],
                "F": [
                    "emotion", "values", "harmony", "subjective", "people", "relationships",
                    "empathy", "compassion", "caring", "sensitive", "warm", "nurturing",
                    "personal", "feelings", "heart", "understanding", "supportive"
                ]
            },
            "judging_perceiving": {
                "J": [
                    "structure", "planning", "organized", "closure", "decisions", "deadlines",
                    "systematic", "methodical", "orderly", "disciplined", "focused",
                    "goal-oriented", "efficient", "productive", "reliable", "consistent"
                ],
                "P": [
                    "flexibility", "spontaneity", "adaptable", "options", "openness", "exploration",
                    "spontaneous", "free-spirited", "curious", "experimental", "versatile",
                    "open-minded", "exploratory", "adventurous", "creative", "innovative"
                ]
            }
        }
        
        # Sentiment and language complexity indicators
        self.complexity_indicators = [
            "however", "although", "nevertheless", "furthermore", "moreover",
            "consequently", "therefore", "thus", "hence", "subsequently"
        ]
        
        self.abstract_indicators = [
            "concept", "theory", "philosophy", "principle", "paradigm", "framework",
            "perspective", "viewpoint", "approach", "methodology", "strategy"
        ]
    
    def classify_personality(self, text: str) -> Dict:
        """Classify personality type using rule-based analysis."""
        text_lower = text.lower()
        
        # Analyze each dimension
        dimension_scores = self._analyze_dimensions(text_lower)
        
        # Determine MBTI type
        mbti_type = self._construct_mbti_type(dimension_scores)
        
        # Calculate confidence based on score differences
        confidence = self._calculate_confidence(dimension_scores)
        
        return {
            "mbti_type": mbti_type,
            "confidence": confidence,
            "dimensions": dimension_scores,
            "method": "rule-based",
            "analysis": self._get_detailed_analysis(text_lower, dimension_scores)
        }
    
    def _analyze_dimensions(self, text: str) -> Dict:
        """Analyze each MBTI dimension using keyword matching and linguistic analysis."""
        dimension_results = {}
        
        for dimension, mappings in self.keyword_mappings.items():
            scores = {"I": 0, "E": 0} if "I" in mappings else {"S": 0, "N": 0} if "S" in mappings else {"T": 0, "F": 0} if "T" in mappings else {"J": 0, "P": 0}
            
            # Count keyword occurrences
            for letter, keywords in mappings.items():
                for keyword in keywords:
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = len(re.findall(pattern, text))
                    scores[letter] += matches
            
            # Additional linguistic analysis
            if dimension == "introversion_extraversion":
                scores = self._analyze_social_patterns(text, scores)
            elif dimension == "sensing_intuition":
                scores = self._analyze_abstraction_patterns(text, scores)
            elif dimension == "thinking_feeling":
                scores = self._analyze_emotional_patterns(text, scores)
            elif dimension == "judging_perceiving":
                scores = self._analyze_planning_patterns(text, scores)
            
            dimension_results[dimension] = scores
        
        return dimension_results
    
    def _analyze_social_patterns(self, text: str, scores: Dict) -> Dict:
        """Analyze social interaction patterns."""
        # Count first-person pronouns (indicates introspection)
        i_patterns = len(re.findall(r'\b(i|me|my|myself)\b', text))
        scores["I"] += i_patterns * 0.5
        
        # Count collaborative words
        we_patterns = len(re.findall(r'\b(we|us|our|team|group|together)\b', text))
        scores["E"] += we_patterns * 0.5
        
        return scores
    
    def _analyze_abstraction_patterns(self, text: str, scores: Dict) -> Dict:
        """Analyze abstraction vs concrete thinking patterns."""
        # Count abstract concept words
        abstract_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                           for word in self.abstract_indicators)
        scores["N"] += abstract_count * 0.5
        
        # Count complexity indicators
        complexity_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                             for word in self.complexity_indicators)
        scores["N"] += complexity_count * 0.3
        
        return scores
    
    def _analyze_emotional_patterns(self, text: str, scores: Dict) -> Dict:
        """Analyze emotional vs logical patterns."""
        # Use sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Emotional words
        emotional_words = ["feel", "emotion", "heart", "love", "care", "worry", "hope", "dream"]
        emotional_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                            for word in emotional_words)
        scores["F"] += emotional_count * 0.5
        
        # Logical words
        logical_words = ["logic", "reason", "analyze", "evidence", "fact", "data", "prove"]
        logical_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                           for word in logical_words)
        scores["T"] += logical_count * 0.5
        
        return scores
    
    def _analyze_planning_patterns(self, text: str, scores: Dict) -> Dict:
        """Analyze planning vs flexibility patterns."""
        # Planning words
        planning_words = ["plan", "schedule", "organize", "structure", "deadline", "goal"]
        planning_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                           for word in planning_words)
        scores["J"] += planning_count * 0.5
        
        # Flexibility words
        flexible_words = ["flexible", "adapt", "spontaneous", "explore", "try", "experiment"]
        flexible_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text)) 
                           for word in flexible_words)
        scores["P"] += flexible_count * 0.5
        
        return scores
    
    def _construct_mbti_type(self, dimension_scores: Dict) -> str:
        """Construct MBTI type from dimension scores."""
        mbti_type = ""
        
        # Map dimensions to MBTI letters
        dimension_mapping = {
            "introversion_extraversion": ("I", "E"),
            "sensing_intuition": ("S", "N"),
            "thinking_feeling": ("T", "F"),
            "judging_perceiving": ("J", "P")
        }
        
        for dimension, (letter1, letter2) in dimension_mapping.items():
            scores = dimension_scores[dimension]
            if scores[letter1] > scores[letter2]:
                mbti_type += letter1
            elif scores[letter2] > scores[letter1]:
                mbti_type += letter2
            else:
                # Tie-breaker: use common defaults
                defaults = {"I": "I", "S": "S", "T": "T", "J": "J"}
                mbti_type += defaults[letter1]
        
        return mbti_type
    
    def _calculate_confidence(self, dimension_scores: Dict) -> float:
        """Calculate confidence based on score differences."""
        total_confidence = 0
        dimension_count = 0
        
        for dimension, scores in dimension_scores.items():
            if len(scores) == 2:
                values = list(scores.values())
                if sum(values) > 0:  # Avoid division by zero
                    # Calculate how clear the preference is
                    max_score = max(values)
                    min_score = min(values)
                    total_score = sum(values)
                    
                    if total_score > 0:
                        clarity = (max_score - min_score) / total_score
                        total_confidence += clarity
                        dimension_count += 1
        
        return total_confidence / dimension_count if dimension_count > 0 else 0.5
    
    def _get_detailed_analysis(self, text: str, dimension_scores: Dict) -> Dict:
        """Get detailed analysis of the classification."""
        analysis = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(sent_tokenize(text)),
            "sentiment": self.sentiment_analyzer.polarity_scores(text),
            "dimension_analysis": {}
        }
        
        for dimension, scores in dimension_scores.items():
            total_score = sum(scores.values())
            if total_score > 0:
                percentages = {k: (v / total_score) * 100 for k, v in scores.items()}
                analysis["dimension_analysis"][dimension] = {
                    "scores": scores,
                    "percentages": percentages,
                    "preference": max(scores, key=scores.get) if scores else None
                }
        
        return analysis
    
    def get_personality_description(self, mbti_type: str) -> Dict:
        """Get personality description for a given MBTI type."""
        # This would typically load from a file, but for simplicity we'll include basic descriptions
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
        }
        
        return descriptions.get(mbti_type, {
            "name": "Unknown Type",
            "strengths": ["Unknown"],
            "careers": ["Various"],
            "description": "Personality type not fully determined."
        }) 
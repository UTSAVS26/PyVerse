"""
Emotion Analyzer Module
Enhanced emotion detection using Watson NLP with improved error handling and features.
"""

import requests
import json
import time
from typing import Dict, Optional, Union

class EmotionAnalyzer:
    """
    Enhanced emotion analyzer with Watson NLP integration.
    """
    
    def __init__(self):
        self.url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
        self.headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
        self.emotion_emojis = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'joy': '😊',
            'sadness': '😢'
        }
        self.emotion_colors = {
            'anger': '#ff4444',
            'disgust': '#8b4513',
            'fear': '#9932cc',
            'joy': '#ffd700',
            'sadness': '#4169e1'
        }
    
    def analyze_emotion(self, text: str, max_retries: int = 3) -> Optional[Dict[str, Union[str, float]]]:
        """
        Analyze emotions in the provided text with retry logic.
        
        Args:
            text (str): Text to analyze
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Dict containing emotion scores and dominant emotion, or None if analysis fails
        """
        # Handle blank entries
        if not text or text.strip() == "":
            return None
        
        for attempt in range(max_retries):
            try:
                # Prepare the request
                input_json = {"raw_document": {"text": text}}
                
                # Make the API request with increased timeout
                response = requests.post(
                    self.url, 
                    json=input_json, 
                    headers=self.headers,
                    timeout=120  # Increased timeout to 120 seconds (2 minutes)
                )
                
                # Check for bad request
                if response.status_code == 400:
                    print(f"Bad request (400) - Invalid input text")
                    return None
                
                # Check for successful response
                if response.status_code == 200:
                    response_dict = json.loads(response.text)
                    emotions = response_dict['emotionPredictions'][0]['emotion']
                    
                    # Extract emotion scores
                    emotion_scores = {
                        'anger': emotions.get('anger', 0),
                        'disgust': emotions.get('disgust', 0),
                        'fear': emotions.get('fear', 0),
                        'joy': emotions.get('joy', 0),
                        'sadness': emotions.get('sadness', 0)
                    }
                    
                    # Find dominant emotion
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    
                    # Calculate confidence level
                    dominant_score = emotion_scores[dominant_emotion]
                    total_score = sum(emotion_scores.values())
                    confidence = (dominant_score / total_score * 100) if total_score > 0 else 0
                    
                    # Prepare result with additional metadata
                    result = {
                        **emotion_scores,
                        'dominant_emotion': dominant_emotion,
                        'confidence': round(confidence, 2),
                        'emoji': self.emotion_emojis.get(dominant_emotion, '😐'),
                        'color': self.emotion_colors.get(dominant_emotion, '#666666'),
                        'intensity': self._calculate_intensity(dominant_score)
                    }
                    
                    return result
                
                # Handle other status codes
                elif response.status_code == 429:
                    print(f"Rate limit exceeded (429) - Attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                elif response.status_code == 500:
                    print(f"Server error (500) - Attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout as e:
                print(f"Timeout error (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    print("Max retries reached for timeout")
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Wait 3 seconds before retry
                    continue
                else:
                    print("Max retries reached for connection error")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print("Max retries reached for request error")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
        
        return None
    
    def _calculate_intensity(self, score: float) -> str:
        """
        Calculate emotion intensity based on score.
        
        Args:
            score (float): Emotion score
            
        Returns:
            str: Intensity level (low, medium, high)
        """
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def get_emotion_description(self, emotion: str) -> str:
        """
        Get a description for the given emotion.
        
        Args:
            emotion (str): Emotion name
            
        Returns:
            str: Description of the emotion
        """
        descriptions = {
            'anger': 'Feeling of strong displeasure or hostility',
            'disgust': 'Feeling of revulsion or strong disapproval',
            'fear': 'Feeling of being afraid or worried about something',
            'joy': 'Feeling of happiness, satisfaction, or contentment',
            'sadness': 'Feeling of sorrow, unhappiness, or depression'
        }
        return descriptions.get(emotion, 'Unknown emotion')
    
    def get_suggestions(self, emotion: str) -> str:
        """
        Get suggestions based on the dominant emotion.
        
        Args:
            emotion (str): Dominant emotion
            
        Returns:
            str: Suggestion for the emotion
        """
        suggestions = {
            'anger': 'Try taking deep breaths or engaging in physical exercise to release tension.',
            'disgust': 'Consider focusing on positive aspects or removing yourself from the situation.',
            'fear': 'Break down your concerns into manageable steps and seek support if needed.',
            'joy': 'Enjoy this positive moment and consider sharing your happiness with others.',
            'sadness': 'Allow yourself to feel these emotions and consider reaching out for support.'
        }
        return suggestions.get(emotion, 'Take time to reflect on your emotions.')

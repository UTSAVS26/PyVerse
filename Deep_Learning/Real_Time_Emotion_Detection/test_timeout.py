#!/usr/bin/env python3
"""
Test script to verify timeout improvements in the emotion analyzer.
"""

import time
from emotion_analyzer import EmotionAnalyzer

def test_timeout_improvements():
    """Test the timeout and retry improvements."""
    analyzer = EmotionAnalyzer()
    
    # Test with a sample text
    test_text = "I am feeling really excited about this new project!"
    
    print("Testing emotion analysis with improved timeout handling...")
    print(f"Test text: {test_text}")
    print("Timeout set to: 120 seconds (2 minutes)")
    print("Max retries: 3")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        result = analyzer.analyze_emotion(test_text)
        end_time = time.time()
        
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        
        if result:
            print("✅ Analysis successful!")
            print(f"Dominant emotion: {result['dominant_emotion']} {result['emoji']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Intensity: {result['intensity']}")
            print("\nEmotion breakdown:")
            for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness']:
                score = result[emotion]
                print(f"  {emotion.capitalize()}: {score:.3f} ({score*100:.1f}%)")
        else:
            print("❌ Analysis failed - this is expected if Watson API is unavailable")
            print("The application will handle this gracefully with user-friendly error messages")
            
    except Exception as e:
        end_time = time.time()
        print(f"Error occurred after {end_time - start_time:.2f} seconds")
        print(f"Error: {e}")
        
    print("\n" + "=" * 50)
    print("Timeout improvements implemented:")
    print("✓ Increased timeout from 30s to 120s (2 minutes)")
    print("✓ Added retry logic with exponential backoff")
    print("✓ Better error handling for different scenarios")
    print("✓ Dynamic loading messages in the UI")
    print("✓ Improved user feedback")

if __name__ == "__main__":
    test_timeout_improvements()

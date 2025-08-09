"""
Enhanced Emotion Detection Web Application
A modern, engaging web interface for real-time emotion analysis using Watson NLP.
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import os
from emotion_analyzer import EmotionAnalyzer

app = Flask(__name__)

# Initialize emotion analyzer
analyzer = EmotionAnalyzer()

# Store analysis history (in production, use a database)
analysis_history = []

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    """
    Analyze emotion from text input.
    
    Returns:
        JSON response with emotion analysis results
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Please enter some text to analyze'
            }), 400
        
        # Analyze emotions
        result = analyzer.analyze_emotion(text)
        
        if result is None or result.get('dominant_emotion') is None:
            return jsonify({
                'success': False,
                'error': 'Unable to analyze the text. The Watson NLP service may be temporarily unavailable. Please try again in a few moments.'
            }), 400
        
        # Add timestamp and original text to result
        result['timestamp'] = datetime.now().isoformat()
        result['original_text'] = text
        result['text_length'] = len(text)
        result['word_count'] = len(text.split())
        
        # Add to history
        analysis_history.append(result)
        
        # Keep only last 10 analyses
        if len(analysis_history) > 10:
            analysis_history.pop(0)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/history')
def get_history():
    """Get analysis history."""
    return jsonify({
        'success': True,
        'history': analysis_history
    })

@app.route('/stats')
def get_stats():
    """Get statistics about emotion analysis."""
    if not analysis_history:
        return jsonify({
            'success': True,
            'stats': {
                'total_analyses': 0,
                'most_common_emotion': None,
                'emotion_distribution': {}
            }
        })
    
    # Calculate statistics
    total_analyses = len(analysis_history)
    emotions = [item['dominant_emotion'] for item in analysis_history]
    
    emotion_distribution = {}
    for emotion in emotions:
        emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
    
    most_common_emotion = max(emotion_distribution, key=emotion_distribution.get)
    
    return jsonify({
        'success': True,
        'stats': {
            'total_analyses': total_analyses,
            'most_common_emotion': most_common_emotion,
            'emotion_distribution': emotion_distribution
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

# Modern Sentiment Analysis Web App

A beautiful, modern web application for sentiment analysis with improved UI/UX design, built with Flask and featuring a robust fallback sentiment analysis system.

## Features

- 🎨 **Modern UI/UX Design**: Clean, responsive interface with smooth animations
- 🧠 **Dual Analysis System**: Primary Skills Network Watson service with intelligent fallback
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- ⚡ **Real-time Feedback**: Instant character counting and input validation
- 🎯 **Visual Results**: Confidence scores with animated progress bars for positive, negative, and neutral sentiment
- 🔄 **Loading States**: Smooth loading animations and user feedback
- 📝 **Sample Texts**: Pre-made examples to test the application quickly
- 🛡️ **Robust Error Handling**: Automatic fallback to local sentiment analysis when external service is unavailable
- 🎭 **Three Sentiment Types**: Positive, Negative, and Neutral with color-coded results

## Screenshots

The app features:
- Gradient background with a modern card-based layout
- Animated buttons and interactive elements
- Color-coded sentiment results (positive/negative)
- Animated confidence score bars
- Character counting with color indicators
- Sample text buttons for quick testing

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd sentiment-analysis
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **No additional setup required!**
   - The application uses a fallback sentiment analysis system that works without external API keys
   - Primary service: Skills Network Watson (automatic fallback if unavailable)
   - Secondary service: Local rule-based sentiment analysis (always available)

## Usage

1. **Run the application:**
```bash
python app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:5000
```

3. **Use the application:**
   - Enter text in the textarea
   - Click "Analyze Sentiment" button
   - View the results with confidence scores
   - Try sample texts for quick testing

## Technical Details

### Backend
- **Flask**: Lightweight web framework for Python
- **Skills Network Watson**: Primary sentiment analysis service
- **Local Fallback**: Rule-based sentiment analysis for reliability
- **JSON Processing**: Efficient data handling and parsing
- **Error Handling**: Robust exception management with automatic fallback

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Advanced styling with gradients, animations, and flexbox
- **JavaScript**: Modern ES6+ features with async/await
- **Font Awesome**: Icons for better visual appeal
- **Google Fonts**: Inter font family for modern typography

### Features Implemented
- Character counting with color coding
- Auto-resizing textarea
- Loading states and animations
- Error handling with user-friendly messages
- Responsive design for all screen sizes
- Sample text functionality
- Keyboard shortcuts (Ctrl+Enter to analyze)
- Celebration animations for high confidence results
- Three-tier sentiment classification (Positive, Negative, Neutral)
- Automatic service fallback for maximum reliability

## API Endpoints

- `GET /`: Main application page
- `POST /analyze`: Sentiment analysis endpoint
  - Input: `textToAnalyze` (form data)
  - Output: JSON with `label` and `score`

## Customization

### Styling
Modify `static/style.css` to change:
- Color schemes
- Typography
- Layout and spacing
- Animations and transitions

### Functionality
Modify `static/script.js` to add:
- New sample texts
- Additional validation
- Enhanced animations
- New features

### Backend
Modify `app.py` to:
- Use different models
- Add new endpoints
- Implement additional features

## Model Information

### Primary Service: Skills Network Watson
- **Service**: Skills Network Watson Sentiment Analysis
- **Type**: BERT-based sentiment analysis model
- **Labels**: positive, negative, neutral
- **Confidence**: Score between 0 and 1 (confidence level)

### Fallback Service: Local Rule-Based Analysis
- **Type**: Dictionary-based sentiment analysis
- **Method**: Positive/negative word counting with weighted scoring
- **Labels**: POSITIVE, NEGATIVE, NEUTRAL
- **Confidence**: Calculated based on word frequency and sentiment strength
- **Reliability**: Always available, no network dependency

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers

## Performance Notes

- **Primary Service**: Fast API responses from Skills Network Watson service
- **Fallback Service**: Instant local analysis with no network dependency
- **Reliability**: 100% uptime with automatic fallback system
- **Accuracy**: High accuracy from BERT-based primary model, reasonable accuracy from rule-based fallback
- **No API Limits**: Free to use with no usage restrictions

## Project Structure

```
sentiment-analysis/
├── app.py                    # Main Flask application
├── sentiment_analysis.py     # Sentiment analysis logic with fallback
├── test_sentiment.py        # Test script for validation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── DEPLOYMENT.md           # Deployment guide
├── QUICK_START.md          # Quick start guide
├── Procfile                # Deployment configuration
├── runtime.txt             # Python version specification
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore file
├── templates/
│   └── index.html          # Main HTML template
└── static/
    ├── style.css           # CSS styling
    └── script.js           # JavaScript functionality
```

## Testing

Run the included test script to verify functionality:
```bash
python test_sentiment.py
```

This will test various text inputs and show the sentiment analysis results.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Deployment Options

See the **DEPLOYMENT.md** file for detailed instructions on hosting this application for free on various platforms.

## Acknowledgments

- Skills Network for the Watson sentiment analysis service
- Flask community for the excellent web framework
- Font Awesome for beautiful icons
- Google Fonts for modern typography
- Python community for the amazing ecosystem

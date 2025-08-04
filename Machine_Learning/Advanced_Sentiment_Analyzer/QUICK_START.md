# Quick Start Guide 🚀

Get your Sentiment Analysis app running in 5 minutes!

## Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5000`

## Test the App

### Run Tests
```bash
python test_sentiment.py
```

### Try These Examples:
- **Positive**: "I love this product! It's amazing!"
- **Negative**: "This is terrible and disappointing."
- **Neutral**: "The weather is okay today."

## Deploy Online (Free)

### Option 1: Render (Recommended)
1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new Web Service
4. Connect your GitHub repo
5. Deploy automatically!

### Option 2: Railway
1. Install Railway CLI: `npm install -g @railway/cli`
2. Run: `railway login`
3. Run: `railway new`
4. Select GitHub repo
5. Deploy: `railway up`

### Option 3: Heroku
1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`

## Files Overview

- `app.py` - Main Flask application
- `sentiment_analysis.py` - Sentiment analysis logic
- `templates/index.html` - Web interface
- `static/` - CSS and JavaScript files
- `requirements.txt` - Python dependencies
- `Procfile` - Deployment configuration

## Need Help?

- Check `README.md` for detailed information
- Check `DEPLOYMENT.md` for deployment guides
- Run `python test_sentiment.py` to verify everything works

## Features

✅ **Works offline** - No API keys needed!  
✅ **Fallback system** - Always reliable  
✅ **Modern UI** - Beautiful and responsive  
✅ **Three sentiments** - Positive, Negative, Neutral  
✅ **Free deployment** - Multiple platform options  

Happy analyzing! 🎉

# Deployment Guide - Free Hosting Options

This guide provides step-by-step instructions for deploying your Sentiment Analysis Web App on various free hosting platforms.

## 🚀 Quick Deployment Options

### 1. **Render** (Recommended - Easy & Free)

**Why Render?**
- Free tier with 750 hours/month
- Automatic deployments from GitHub
- Built-in SSL certificates
- Simple setup process

**Steps:**
1. **Prepare your project:**
   ```bash
   # Add a Procfile for Render
   echo "web: python app.py" > Procfile
   
   # Update app.py to use environment PORT
   # Add this to the end of app.py before if __name__ == "__main__":
   import os
   PORT = int(os.environ.get('PORT', 5000))
   
   # Change the last line to:
   app.run(host='0.0.0.0', port=PORT, debug=False)
   ```

2. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/sentiment-analysis.git
   git push -u origin main
   ```

3. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: sentiment-analysis
     - **Region**: Choose closest to you
     - **Branch**: main
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`
   - Click "Create Web Service"

**URL**: Your app will be available at `https://sentiment-analysis-xxxx.onrender.com`

---

### 2. **Railway** (Fast & Developer-Friendly)

**Why Railway?**
- $5 credit monthly (more than enough for small apps)
- Automatic deployments
- Built-in databases if needed
- Simple CLI tool

**Steps:**
1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and deploy:**
   ```bash
   railway login
   railway new
   # Choose "Deploy from GitHub repo"
   # Select your sentiment-analysis repository
   railway up
   ```

3. **Set environment variables (if needed):**
   ```bash
   railway variables set PORT=8080
   ```

**URL**: Your app will be available at `https://sentiment-analysis-production.up.railway.app`

---

### 3. **Heroku** (Popular Choice)

**Why Heroku?**
- Well-established platform
- Free tier (with limitations)
- Great documentation
- Add-ons ecosystem

**Steps:**
1. **Install Heroku CLI:**
   - Download from [heroku.com/cli](https://devcenter.heroku.com/articles/heroku-cli)

2. **Prepare your project:**
   ```bash
   # Create Procfile
   echo "web: python app.py" > Procfile
   
   # Create runtime.txt (optional)
   echo "python-3.10.0" > runtime.txt
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create sentiment-analysis-yourname
   git push heroku main
   ```

**URL**: Your app will be available at `https://sentiment-analysis-yourname.herokuapp.com`

---

### 4. **PythonAnywhere** (Python-Specific)

**Why PythonAnywhere?**
- Free tier specifically for Python apps
- Easy Flask deployment
- Built-in Python environment
- No credit card required

**Steps:**
1. **Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)**

2. **Upload your files:**
   - Use the Files tab to upload your project
   - Or clone from GitHub in a Bash console

3. **Configure Web App:**
   - Go to Web tab
   - Click "Add a new web app"
   - Choose "Flask"
   - Set source code path to your app directory
   - Set WSGI configuration file to point to your app.py

4. **Reload your web app**

**URL**: Your app will be available at `https://yourusername.pythonanywhere.com`

---

### 5. **Vercel** (For Static + Serverless)

**Why Vercel?**
- Excellent for serverless functions
- Fast deployments
- Great for modern web apps
- Free tier with good limits

**Steps:**
1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Create vercel.json:**
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

**URL**: Your app will be available at `https://sentiment-analysis-xxxx.vercel.app`

---

## 📝 Pre-Deployment Checklist

Before deploying to any platform:

1. **Update app.py for production:**
   ```python
   import os
   
   # At the end of app.py
   if __name__ == "__main__":
       port = int(os.environ.get('PORT', 5000))
       app.run(host='0.0.0.0', port=port, debug=False)
   ```

2. **Create/update requirements.txt:**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Test locally:**
   ```bash
   python app.py
   ```

4. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   ```

---

## 🔧 Platform-Specific Files

### For Heroku/Render:
```bash
# Procfile
web: python app.py

# runtime.txt (optional)
python-3.10.0
```

### For Railway:
```bash
# railway.json (optional)
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "python app.py"
  }
}
```

### For Vercel:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

---

## 🌟 Recommended Choice: Render

**For beginners, we recommend Render because:**
- Completely free for personal projects
- Easy GitHub integration
- Automatic deployments
- Built-in SSL
- No credit card required
- Good performance

**Quick Render Deployment:**
1. Push code to GitHub
2. Connect GitHub to Render
3. Deploy automatically
4. Get HTTPS URL instantly

---

## 💡 Tips for Success

1. **Test locally first** - Always ensure your app works locally before deploying
2. **Use environment variables** - For any sensitive data (though this app doesn't need any)
3. **Monitor logs** - Check deployment logs if something goes wrong
4. **Keep it simple** - Start with the basic deployment, then add features
5. **Read the docs** - Each platform has excellent documentation

---

## 🆘 Troubleshooting Common Issues

### Port Issues:
```python
# Fix: Use environment PORT variable
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### Module Not Found:
```bash
# Fix: Update requirements.txt
pip freeze > requirements.txt
```

### Build Failures:
```bash
# Fix: Check Python version compatibility
# Most platforms support Python 3.8-3.11
```

### Static Files Not Loading:
```python
# Fix: Use proper Flask static file serving
app = Flask(__name__)
# Flask automatically serves static files from /static/
```

---

## 🎉 After Deployment

1. **Test your live app** - Check all functionality works
2. **Share the URL** - Your app is now live and accessible worldwide!
3. **Monitor usage** - Keep an eye on platform usage limits
4. **Update when needed** - Push changes to trigger redeployment

---

**Need help?** Most platforms have excellent documentation and community support. Check their official docs for the most up-to-date instructions!

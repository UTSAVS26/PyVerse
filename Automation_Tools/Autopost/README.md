# Autopost - DEV.to Auto-Poster

This tool automatically generates comprehensive technical blog posts using Google's Gemini AI and posts them to DEV.to. It creates detailed articles with proper formatting, code examples, and structured content about the latest technology trends.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API Keys

#### Gemini API Key

1. Go to [Google Makersuite API Key page](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`

#### DEV.to API Key (Primary Method)

1. Go to [DEV.to Extensions Settings](https://dev.to/settings/extensions)
2. Generate a new API key
3. Add it to your `.env` file: `DEVTO_API_KEY=your_key_here`

### 3. Environment File (.env)

Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here1
DEVTO_API_KEY=your_devto_api_key_here
```

## Usage

### Standard Usage (API Method)
```bash
python main.py
```

### Debug Mode
```bash
python main.py --debug
```

## Troubleshooting

### Common Issues

1. **Rate Limit Error (429)**
   - DEV.to limits posts to prevent spam
   - Wait 5 minutes (300 seconds) between posts
   - Error message: "Rate limit reached, try again in 300 seconds"

2. **API Key Issues**
   - Make sure your API keys are correct in the `.env` file
   - For Gemini: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
   - For DEV.to: <https://dev.to/settings/extensions>

3. **Content Generation Issues**
   - Check your Gemini API key is valid
   - Ensure you have internet connection
   - The script uses gemini-1.5-flash model

4. **"Title has already been used" Error**
   - DEV.to prevents duplicate titles within 5 minutes
   - The script generates unique titles automatically
   - Wait a few minutes and try again

### Files Generated During Debugging

- Debug files are automatically cleaned up after successful runs
- Only essential project files remain in the directory

## Features

- **Rich Content Generation**: Creates comprehensive technical blog posts with:
  - Catchy, unique titles
  - Introduction and conclusion sections
  - Multiple content sections with subheadings
  - Code examples and technical details
  - Proper markdown formatting
- **DEV.to Integration**: Posts directly using DEV.to API
- **Automatic Error Handling**: Robust error handling and recovery
- **Rate Limit Awareness**: Handles DEV.to posting restrictions
- **Content Variety**: Covers AI, programming frameworks, cybersecurity, cloud computing

## What Gets Generated

The script creates detailed technical articles covering topics like:

- Generative AI advancements
- New programming frameworks
- Cybersecurity developments
- Cloud computing trends
- Emerging technologies

Each post includes:

- Professional title and structure
- Code examples where relevant
- Technical insights and analysis
- Developer-focused content

## Files

- `main.py` - Main script (API-only version)
- `gemini_client.py` - Gemini AI integration with enhanced prompts
- `devto_api_client.py` - DEV.to API client
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create this yourself)

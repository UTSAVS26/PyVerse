import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_technical_post():
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = {
        "contents": [
            {
                "parts": [
                    {
                        "text": """Write a comprehensive technical blog post about recent developments in technology. 
                        Include:
                        - A catchy, unique title starting with #
                        - Introduction paragraph
                        - 2-3 main sections with subheadings
                        - Code examples or technical details where relevant
                        - Conclusion
                        
                        Topics could include: AI developments, new programming frameworks, cybersecurity, cloud computing, or emerging technologies.
                        Make it informative and engaging for developers.
                        Format it as markdown for DEV.to."""
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(url, json=prompt)
        response.raise_for_status()
        data = response.json()
        
        if 'candidates' in data and len(data['candidates']) > 0:
            text = data['candidates'][0]['content']['parts'][0]['text']
            return text
        else:
            return "# Tech Update\n\nFailed to generate post content. Please check your API key."
            
    except requests.exceptions.RequestException as e:
        return f"# Tech Update\n\nAPI request failed: {str(e)}"
    except Exception as e:
        return f"# Tech Update\n\nError generating content: {str(e)}"

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEVTO_API_KEY = os.getenv("DEVTO_API_KEY")

def post_to_devto_api(title, body, tags=None):
    """
    Post an article to DEV.to using their API
    Requires a DEV.to API key to be set in the .env file
    """
    if not DEVTO_API_KEY:
        raise ValueError("DEVTO_API_KEY must be set in .env file. Get it from https://dev.to/settings/extensions")
    
    url = "https://dev.to/api/articles"
    headers = {
        "api-key": DEVTO_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Default tags if none provided
    if tags is None:
        tags = ["technology", "programming", "news"]
    
    data = {
        "article": {
            "title": title,
            "body_markdown": body,
            "published": True,
            "tags": tags
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        article_data = response.json()
        print(f"Article published successfully!")
        print(f"URL: {article_data.get('url')}")
        return article_data
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise Exception("Invalid API key. Please check your DEVTO_API_KEY in .env file")
        elif response.status_code == 422:
            error_details = response.json()
            raise Exception(f"Validation error: {error_details}")
        else:
            raise Exception(f"HTTP error {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"Error posting to DEV.to API: {e}")

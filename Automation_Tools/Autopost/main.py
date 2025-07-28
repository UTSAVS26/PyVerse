import os
import sys
from dotenv import load_dotenv

try:
    from gemini_client import generate_technical_post
    from devto_api_client import post_to_devto_api
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure gemini_client.py and devto_api_client.py exist in the same directory")
    exit(1)

load_dotenv()

def split_title_and_body(post):
    lines = post.split('\n')
    title = lines[0].replace("# ", "").strip() if lines[0].startswith("#") else f"Tech Update {hash(post) % 10000}"
    body = "\n".join(lines[1:]).strip()
    return title, body

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_actual_gemini_api_key_here":
        print("Error: Please set your actual Gemini API key in the .env file")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        exit(1)
    
    # Check if we should run in debug mode (non-headless)
    debug_mode = "--debug" in sys.argv
    use_api = True  # Only use API method since browser automation is not available
    
    if debug_mode:
        print("Running in debug mode")
    
    print("Using DEV.to API method")
    if not os.getenv("DEVTO_API_KEY"):
        print("Error: DEVTO_API_KEY not found. Get your API key from https://dev.to/settings/extensions")
        print("Add DEVTO_API_KEY=your_key_here to your .env file")
        exit(1)
    
    try:
        print("Generating technical post...")
        post = generate_technical_post()
        print("Post generated successfully!")
        
        title, body = split_title_and_body(post)
        print(f"Title: {title}")
        print(f"Body preview: {body[:100]}...")
        
        print("Posting to DEV.to...")
        post_to_devto_api(title, body)
        print("Post successfully uploaded to DEV.to!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your API keys and internet connection")

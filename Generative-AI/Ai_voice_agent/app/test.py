import requests

url = "http://127.0.0.1:8000/tts"
data = {"text": "Hello, I am your AI voice assistant."}

response = requests.post(url, json=data)
print(response.json())

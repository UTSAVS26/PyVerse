import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_intent(text: str) -> str:
    prompt = f"Extract the customer support intent from the following message:\n\n{text}\n\nIntent:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip()

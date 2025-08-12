import together
import os
from dotenv import load_dotenv

load_dotenv()

def scientist_response(topic: str, round_num: int, memory: dict) -> str:
    prompt = f"""You are a Scientist in a debate on: "{topic}".
The debate has 8 rounds. Each round, you and the Philosopher take turns.
Keep responses under 40 words.

Round {round_num} response:"""

    response = together.Complete.create(
        prompt=prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens=60,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    return response['choices'][0]['text'].strip()

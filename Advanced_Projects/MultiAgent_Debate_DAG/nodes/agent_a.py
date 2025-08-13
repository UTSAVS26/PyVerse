import re 
import together
import os
from dotenv import load_dotenv

load_dotenv()

together_api_key = os.getenv("TOGETHER_API_KEY")
if together_api_key:
    # Make auth explicit; Together SDK will use this value if not already configured.
    together.api_key = together_api_key


def scientist_response(topic: str, round_num: int, memory: dict) -> str:
    # Include recent context from memory (last ~6 lines).
    transcript = memory.get("transcript", "") if isinstance(memory, dict) else ""
    context_section = ""
    if transcript:
        last_lines = transcript.splitlines()[-6:]
        context_section = "Context so far (recent turns):\n" + "\n".join(last_lines) + "\n"

    prompt = f"""You are a Scientist in a debate on: "{topic}".
The debate has 8 rounds. Each round, you and the Scientist take turns.
Respond in ONE line (<= 40 words). Do NOT include any role labels (no 'Scientist:' or 'Philosopher:').
{context_section}
Round {round_num} response:"""

    try:
        response = together.Complete.create(
            prompt=prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=60,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
    except Exception as e:
        raise RuntimeError(f"Together API call failed for Scientist (round {round_num}): {e}") from e

    if not isinstance(response, dict) or "choices" not in response or not response["choices"]:
        raise ValueError(f"Unexpected Together response format: {response!r}")
    first = response["choices"][0]
    text = (first.get("text") or "").strip()
    if not text:
        raise ValueError(f"Empty text in Together response: {response!r}")
    return text
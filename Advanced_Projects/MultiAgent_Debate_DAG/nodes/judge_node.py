import re
import together
import os
from dotenv import load_dotenv

load_dotenv()

# Fail fast and set Together API key explicitly
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise RuntimeError(
        "TOGETHER_API_KEY not found in environment. "
        "Create a .env with TOGETHER_API_KEY=<your_key> or export it before running."
    )
together.api_key = api_key

def judge_debate(topic, memory):
    transcript = memory.get('transcript', '') if isinstance(memory, dict) else ''
    # Keep prompt concise: use only the recent portion of the debate
    if transcript:
        lines = transcript.splitlines()
        transcript = "\n".join(lines[-80:])  # window of recent lines

    prompt = f"""You are the judge of an 8-round debate between a Scientist and a Philosopher on the topic: "{topic}".

Debate transcript:
{transcript}

Instructions:
- Provide a 1-line summary.
- Declare the winner clearly.
- Give a short reason.

Format exactly:
[Judge] Summary: ...
[Judge] Winner: ...
Reason: ...
"""

    try:
        response = together.Complete.create(
            prompt=prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        # Handle dict- or object-like responses defensively
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        first = choices[0] if choices else None
        text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
        output = (text or "").strip()
    except Exception as e:
        # Degrade gracefully on API/network errors
        return "", "", f"Judge call failed: {e}"

    # Parse lines to avoid duplication
    lines = output.splitlines()
    summary_line = next((line for line in lines if line.strip().startswith("[Judge] Summary")), "")
    winner_line = next((line for line in lines if line.strip().startswith("[Judge] Winner")), "")
    reason_line = next((line for line in lines if line.strip().startswith("Reason")), "")
    summary_line = summary_line.split(":", 1)[1].strip() if ":" in summary_line else summary_line.strip()
    winner_line = winner_line.split(":", 1)[1].strip() if ":" in winner_line else winner_line.strip()
    reason_line = reason_line.split(":", 1)[1].strip() if ":" in reason_line else reason_line.strip()

    # Normalize winner to expected values
    wl = winner_line.lower()
    if "scientist" in wl:
        winner_line = "Scientist"
    elif "philosopher" in wl:
        winner_line = "Philosopher"
    elif "tie" in wl or "draw" in wl:
        winner_line = "Tie"
    elif not winner_line:
        winner_line = "Undecided"

    # Provide minimal defaults if the model omits fields
    if not summary_line:
        summary_line = "Debate concluded."
    if not reason_line:
        reason_line = "Insufficient data to determine a reason."

    return summary_line, winner_line, reason_line

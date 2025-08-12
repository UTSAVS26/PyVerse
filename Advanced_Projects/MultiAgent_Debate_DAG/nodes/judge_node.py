import together
import os
from dotenv import load_dotenv

load_dotenv()

def judge_debate(topic, memory):
    transcript = memory.get('transcript', '')

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

    response = together.Complete.create(
        prompt=prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    # Parse the output string cleanly
    output = response['choices'][0]['text'].strip()

    # Parse lines to avoid duplication
    summary_line = next((l for l in output.split('\n') if l.startswith("[Judge] Summary:")), "").replace("[Judge] Summary:", "").strip()
    winner_line = next((l for l in output.split('\n') if l.startswith("[Judge] Winner:")), "").replace("[Judge] Winner:", "").strip()
    reason_line = next((l for l in output.split('\n') if l.startswith("Reason:")), "").replace("Reason:", "").strip()

    return summary_line, winner_line, reason_line
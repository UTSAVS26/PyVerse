def update_memory(memory, round_str, _response):
    # Append current round to the transcript
    current_transcript = memory.get("transcript", "")
    
    if current_transcript:
        new_transcript = current_transcript + "\n" + round_str
    else:
        new_transcript = round_str

    memory["transcript"] = new_transcript
    return memory
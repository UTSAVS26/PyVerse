def get_debate_topic():
    """
    Obtain the debate topic from the environment or interactive prompt.
    - If DEBATE_TOPIC is set, use it.
    - Otherwise, prompt the user.
    - Whitespace is trimmed and empty input is rejected.
    """
    import os
    try:
        env_topic = os.getenv("DEBATE_TOPIC")
        if env_topic:
            topic = env_topic.strip()
        else:
            topic = input("Enter topic for debate: ").strip()
    except (EOFError, KeyboardInterrupt) as e:
        raise RuntimeError(
            "No interactive input available. Set DEBATE_TOPIC to run non-interactively."
        ) from e
    if not topic:
        raise ValueError("Debate topic cannot be empty.")
    return topic
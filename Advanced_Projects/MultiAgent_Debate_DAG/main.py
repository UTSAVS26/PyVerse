from nodes.user_input import get_debate_topic
from nodes.agent_a import scientist_response
from nodes.agent_b import philosopher_response
from nodes.memory_node import update_memory
from nodes.judge_node import judge_debate
import time
import os
from datetime import datetime


def main():
    topic = get_debate_topic()
    memory = {"topic": topic, "transcript": "", "round": 1}
    log_lines = []  # to collect all messages

    
    print(f"\nStarting debate between Scientist and Philosopher...")

    # Run exactly 8 rounds (4 each)
    for i in range(8):
        try:
            if i % 2 == 0:
                speaker = "Scientist"
                response = scientist_response(topic, memory['round'], memory)
            else:
                speaker = "Philosopher"
                response = philosopher_response(topic, memory['round'], memory)
        except Exception as e:
            speaker = "Unknown"
            response = f"(error fetching response: {e})"

        # Clean single-line output per round (avoid IndexError on empty responses)
        first_line = (response or "").strip().splitlines()[0] if response else ""
        round_str = f"[Round {memory['round']}] {speaker}: {first_line}"
        print(round_str)

        # Log this round
        log_lines.append(round_str)

        # Update memory
        memory = update_memory(memory, round_str, first_line)
        memory['round'] += 1
        time.sleep(0.3)

    print()  # Single gap before judge output

    # Final judgment
    try:
        summary, winner, reason = judge_debate(topic, memory)
    except Exception as e:
        summary, winner, reason = "", "", f"Judging failed: {e}"
    print(f"[Judge] Summary of debate: {summary.strip()}")
    print(f"[Judge] Winner: {winner.strip()}")
    print(f"Reason: {reason.strip()}")
    
    # Log judge results
    log_lines.append(f"\n[Judge] Summary of debate: {summary.strip()}")
    log_lines.append(f"[Judge] Winner: {winner.strip()}")
    log_lines.append(f"Reason: {reason.strip()}")

    # Save full log to a file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"debate_chat_log.txt"
    log_path = os.path.join(os.getcwd(), log_filename)

    with open(log_path, "w") as f:
        f.write("Enter topic for debate: " + topic + "\n\n")
        f.write("Starting debate between Scientist and Philosopher...\n")
        f.write("\n".join(log_lines))

    print(f"\nChat log saved.")


if __name__ == "__main__":
    main()
import argparse
from core.parser import QueryParser
from core.command_map import CommandMapper
from core.executor import CommandExecutor
from core.utils import log_command


def main():
    parser = argparse.ArgumentParser(description='SmartCLI: Natural Language Command-Line Toolkit')
    parser.add_argument('query', type=str, nargs='+', help='Natural language command')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI API for NLP parsing (requires OPENAI_API_KEY)')
    args = parser.parse_args()
    nl_query = ' '.join(args.query)

    qp = QueryParser(use_openai=args.openai)
    cm = CommandMapper()
    ce = CommandExecutor()

    parsed = qp.parse_query(nl_query)
    command = cm.map_intent(parsed)
    print(f"\n[SmartCLI] Parsed Command:")
    print(f"  Query: {parsed['raw']}")
    print(f"  Intent: {parsed['intent']}")
    print(f"  Entities: {parsed['entities']}")
    print(f"  Shell: {command}\n")
    # Execute with safety checks and user confirmation
    result = ce.execute(command, preview=True)
    
    if result == 0:
        log_command(command, True)
        print("[i] Command executed and logged to smartcli.log.")
    elif result == -1:
        log_command(command, False)
        print("[i] Blocked/failed command logged to smartcli.log.")
    else:
        # User cancelled
        log_command(command, False)
        print("[i] Cancelled command logged to smartcli.log.")

if __name__ == '__main__':
    main() 
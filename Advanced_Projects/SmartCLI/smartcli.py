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
    approved = ce.is_safe(command)
    if approved:
        confirm = input("\u2705 Proceed? [Y/n] ").strip().lower()
        if confirm in ('y', 'yes', ''):
            ce.execute(command, preview=False)
            log_command(command, True)
            print("[i] Command executed and logged to smartcli.log.")
        else:
            print("[i] Command execution cancelled.")
            log_command(command, False)
    else:
        print("[!] Command blocked for safety.")
        log_command(command, False)
        print("[i] Blocked command was logged to smartcli.log.")

if __name__ == '__main__':
    main() 
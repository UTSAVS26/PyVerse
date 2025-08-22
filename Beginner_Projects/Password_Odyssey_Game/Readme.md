# Password Challenge

A Python-based game where players create a password.

## Overview
The Password Challenge is an interactive game that tests your ability to craft a password satisfying specific rules. These rules cover length, character types, specific sequences, and patterns. The game provides feedback on failed conditions to guide improvement.

## Features
- **Unique Conditions**: Rules include length, letter case, digits, special characters, substrings, and patterns.
- **Interactive Feedback**: Displays progress (passed checks) and the first failed condition.
- **Quit Option**: Exit by typing 'quit'.
- **Structured Code**: Uses modular functions for condition generation and password checking.

## Requirements
- Python 3.x
- Standard libraries: `string`, `re`

## Installation
1. Clone or download the repository.
2. Ensure Python 3.x is installed.
3. Run the script:
   ```bash
   python game.py
   ```

## How to Play
1. Run the script.
2. Enter a password when prompted.
3. Review feedback on failed conditions.
4. Adjust the password and retry until all conditions are met or type 'quit' to exit.

## Example Password
An example password that satisfy all 100 conditions:
```
baf1gSUNhiejklmopr5stu$w9c#w!d101b*w934ww0w2
```
*Note*: This is a sample; you must craft your own password to meet all rules.

## Code Structure
- **generate_conditions()**: Defines password rules using lambda functions and regex.
- **check_password(password, conditions)**: Evaluates the password against all conditions, returning success status, failed conditions, and passed count.
- **password_game()**: Runs the interactive game loop.

## Rules Summary
- **Length**: 32–64 characters, divisible by 2.
- **Letters**: At least 15 letters, 3 uppercase, 10 lowercase, 20 unique characters.
- **Digits**: 8 digits, specific counts (e.g., three '1's, two '0's).
- **Special Characters**: Exactly 4 (`#`, `!`, `*`, `$`).
- **Positional**: Specific characters at positions (e.g., starts with lowercase consonant).
- **Substrings**: Must include 'SUN', '101'; exclude 'love', 'pass', etc.
- **Patterns**: No consecutive digits, vowels, or special characters; no palindromes.
- **Math**: ASCII sum of letters even, digit sum divisible by 5.

## Notes
- The game stops at the first failed condition for feedback.
- Conditions are strict; partial matches don't count.
- Test carefully to meet all rules.

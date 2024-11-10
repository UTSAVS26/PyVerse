# Blockchain Vulnerabilities: Formalistic Occurrences

## Introduction
This document aims to provide a formal overview of common vulnerabilities in blockchain technology, focusing on timestamp dependency, reentrancy exploitation, and overflow conditions. The examples include Python code for timestamp creation, Solidity code for preventing reentrancy, and addressing overflow conditions. All examples have been tested on the Ganache platform.

## Vulnerabilities

### 1. Timestamp Dependency

**Description**
Timestamp dependency refers to the reliance on block timestamps for critical logic in smart contracts. Block timestamps are provided by miners and can be manipulated within a reasonable range, leading to potential exploitation.

**Occurrence**
- **Condition**: When smart contracts use `block.timestamp` or `now` to perform critical operations such as time-based releases, conditional statements, or randomness.
- **Impact**: Attackers can manipulate the timestamp to gain unfair advantages or trigger unintended contract behavior.

**Example**
Using Python's `datetime` module to create a timestamp:
```python
import datetime

# Create a timestamp for the current time
current_timestamp = datetime.datetime.now().timestamp()
print(f"Current Timestamp: {current_timestamp}")

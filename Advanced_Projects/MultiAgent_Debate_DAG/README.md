# Multi-Agent Debate DAG (LangGraph)

## 🧠 Overview

This project simulates a debate between two AI agents (Scientist vs Philosopher) on a user-given topic using LangGraph structure.

## 🔧 Environment Setup

This project uses a `.env` file to securely manage API keys.

1. Copy the example environment file:
   ```bash
   cp .env.example .env

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
python dag_diagram.py
```

## 📦 Structure

- `nodes/`: All node logic (agents, judge, memory).
- `main.py`: CLI controller for debate execution.
- `dag_diagram.py`: Generates DAG structure image.
- `debate_chat_log.txt`: Stores entire debate and judgment.

## 🗺️ DAG Nodes

- **UserInputNode**: Gets topic.
- **AgentA / AgentB**: Alternate turns.
- **MemoryNode**: Stores state.
- **JudgeNode**: Declares winner.

## 🏁 Output

- Console printout of the judgment.
- `debate_chat_log.txt`: Full conversation.
- `dag_diagram.png`: Visual DAG layout.
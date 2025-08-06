# ⚙️ LightFlow: A Lightweight Parallel Task Pipeline Framework

> A minimal, Pythonic alternative to Airflow — run dependent tasks in parallel using threads or processes, from a simple YAML workflow file.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)](https://github.com/your-repo/lightflow)
[![Coverage](https://img.shields.io/badge/coverage-44%25-brightgreen.svg)](https://github.com/your-repo/lightflow)

---

## 📌 Project Overview

**LightFlow** is a lightweight, dependency-aware parallel task execution framework written in Python. It allows users to define workflows with steps and dependencies via a simple `YAML` or `JSON` file and executes them using multiprocessing or multithreading. It features real-time visualization of the **DAG (Directed Acyclic Graph)**, persistent checkpointing, failure logging, and retry logic.

Ideal for small-scale automation pipelines, CI tasks, ML model workflows, and more — without needing a full Apache Airflow setup.

### 🎯 Why LightFlow?

- **🚀 Simple**: Define workflows in YAML/JSON - no complex setup required
- **⚡ Fast**: Lightweight execution with minimal overhead
- **🔄 Reliable**: Checkpointing and resume capabilities
- **📊 Visual**: Built-in DAG visualization
- **🔧 Flexible**: Plugin architecture for custom task types
- **🐍 Pythonic**: Native Python integration and extensibility

---

## 🎯 Key Features

- 🧩 **Workflow as Code**  
  Define tasks, dependencies, and execution strategies in a YAML or JSON file.

- 🚀 **Parallel Execution**  
  Run independent tasks concurrently using `concurrent.futures` or `multiprocessing`.

- 🔗 **DAG Visualization**  
  Generate and render task graphs using `graphviz` or `networkx`.

- 🛠️ **Failure Recovery & Checkpointing**  
  Save progress, skip completed tasks, and retry failed ones on rerun.

- 📜 **Rich Logging**  
  Task stdout/stderr logs saved individually with timestamps.

- 📦 **Plugin Architecture**  
  Easily define custom task types (e.g., shell commands, Python scripts, HTTP calls, etc.).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd LightFlow

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

1. **Create a workflow file** (`my_workflow.yaml`):
```yaml
workflow_name: my_first_workflow
tasks:
  hello:
    run: echo "Hello, LightFlow!"
    type: shell
    depends_on: []
  world:
    run: echo "World!"
    type: shell
    depends_on: [hello]
settings:
  max_parallel_tasks: 2
```

2. **Run the workflow**:
```bash
python -m lightflow.cli.main run my_workflow.yaml
```

3. **Visualize the DAG**:
```bash
python -m lightflow.cli.main dag --file my_workflow.yaml --output workflow.svg
```

---

## 🔧 Sample Workflow YAML

```yaml
workflow_name: daily_model_pipeline
tasks:
  fetch_data:
    run: python scripts/fetch.py
    type: shell
    depends_on: []
  
  preprocess:
    run: python scripts/clean.py
    type: shell
    depends_on: [fetch_data]
  
  train_model:
    run: python scripts/train.py
    type: shell
    depends_on: [preprocess]
  
  evaluate:
    run: python scripts/eval.py
    type: shell
    depends_on: [train_model]
  
  notify:
    run: curl -X POST https://webhook.site/send
    type: shell
    depends_on: [evaluate]

settings:
  max_parallel_tasks: 3
  retries: 2
  log_dir: logs/
```

---

## 🧩 Tech Stack

| Purpose | Library / Tool |
|---------|----------------|
| Task Execution | multiprocessing, threading, asyncio |
| DAG Parsing | networkx, pygraphviz |
| CLI Interface | click |
| Logging | logging, rich |
| YAML Support | PyYAML |
| Checkpointing | JSON snapshot |

---

## 📂 Project Structure

```
lightflow/
├── engine/
│   ├── executor.py         # Thread/Process/Async manager
│   ├── dag_builder.py      # Builds and validates DAG
│   ├── checkpoint.py       # Saves and loads execution state
│   ├── logger.py           # Rich logging interface
├── parser/
│   └── workflow_loader.py  # Loads YAML/JSON workflow
├── cli/
│   └── main.py             # Entry point for CLI
├── plugins/
│   └── shell_task.py       # Executes shell tasks
│   └── python_task.py      # Executes Python functions
├── visuals/
│   └── dag_viewer.py       # Generates DAG PNG/SVG
├── examples/
│   └── basic_workflow.yaml
├── tests/
│   └── test_*.py           # Unit tests
├── README.md
└── requirements.txt
```

---

## 💻 CLI Commands

### Core Commands

```bash
# Run a workflow
python -m lightflow.cli.main run examples/basic_workflow.yaml

# Validate workflow
python -m lightflow.cli.main validate examples/basic_workflow.yaml

# Visualize DAG
python -m lightflow.cli.main dag --file examples/basic_workflow.yaml --output dag.svg

# Show task logs
python -m lightflow.cli.main logs --task train_model

# Resume failed workflow
python -m lightflow.cli.main resume examples/basic_workflow.yaml
```

### Utility Commands

```bash
# Create workflow template
python -m lightflow.cli.main template my_workflow

# List checkpoints
python -m lightflow.cli.main list-checkpoints workflow_name

# Clear checkpoints
python -m lightflow.cli.main clear workflow_name
```

### Command Options

```bash
# Run with specific execution mode
python -m lightflow.cli.main run workflow.yaml --mode thread

# Run with custom worker count
python -m lightflow.cli.main run workflow.yaml --max-workers 8

# Dry run (show execution plan without running)
python -m lightflow.cli.main run workflow.yaml --dry-run

# Resume from checkpoint
python -m lightflow.cli.main run workflow.yaml --resume
```

---

## 📊 Examples

### 1. Basic Linear Workflow

```yaml
workflow_name: basic_workflow
tasks:
  fetch_data:
    run: echo "Fetching data..."
    type: shell
    depends_on: []
  
  preprocess:
    run: echo "Preprocessing data..."
    type: shell
    depends_on: [fetch_data]
  
  train_model:
    run: echo "Training model..."
    type: shell
    depends_on: [preprocess]
  
  evaluate:
    run: echo "Evaluating model..."
    type: shell
    depends_on: [train_model]
  
  notify:
    run: echo "Sending notification..."
    type: shell
    depends_on: [evaluate]

settings:
  max_parallel_tasks: 3
  retries: 2
  log_dir: logs/
```

### 2. Python-Based Data Processing

```yaml
workflow_name: python_workflow
tasks:
  setup:
    run: |
      import os
      print("Setting up environment...")
      os.makedirs("data", exist_ok=True)
      print("Environment setup complete!")
    type: python
    depends_on: []
  
  generate_data:
    run: |
      import random
      import json
      
      print("Generating sample data...")
      data = [random.randint(1, 100) for _ in range(10)]
      
      with open("data/sample.json", "w") as f:
          json.dump(data, f)
      
      print(f"Generated {len(data)} data points")
    type: python
    depends_on: [setup]
  
  process_data:
    run: |
      import json
      import statistics
      
      print("Processing data...")
      
      with open("data/sample.json", "r") as f:
          data = json.load(f)
      
      mean = statistics.mean(data)
      median = statistics.median(data)
      std_dev = statistics.stdev(data)
      
      results = {
          "mean": mean,
          "median": median,
          "std_dev": std_dev,
          "count": len(data)
      }
      
      with open("data/results.json", "w") as f:
          json.dump(results, f)
      
      print(f"Processed data - Mean: {mean:.2f}, Median: {median}, Std Dev: {std_dev:.2f}")
    type: python
    depends_on: [generate_data]

settings:
  max_parallel_tasks: 2
  retries: 1
  log_dir: logs/
```

### 3. Parallel Task Execution

```yaml
workflow_name: parallel_workflow
tasks:
  # Initial setup
  setup:
    run: echo "Setting up environment..."
    type: shell
    depends_on: []
  
  # Parallel data processing tasks
  process_data_a:
    run: |
      echo "Processing dataset A..."
      sleep 2
      echo "Dataset A processing complete!"
    type: shell
    depends_on: [setup]
  
  process_data_b:
    run: |
      echo "Processing dataset B..."
      sleep 3
      echo "Dataset B processing complete!"
    type: shell
    depends_on: [setup]
  
  process_data_c:
    run: |
      echo "Processing dataset C..."
      sleep 1
      echo "Dataset C processing complete!"
    type: shell
    depends_on: [setup]
  
  # Parallel model training tasks
  train_model_x:
    run: |
      echo "Training model X..."
      sleep 4
      echo "Model X training complete!"
    type: shell
    depends_on: [process_data_a]
  
  train_model_y:
    run: |
      echo "Training model Y..."
      sleep 3
      echo "Model Y training complete!"
    type: shell
    depends_on: [process_data_b]
  
  train_model_z:
    run: |
      echo "Training model Z..."
      sleep 2
      echo "Model Z training complete!"
    type: shell
    depends_on: [process_data_c]
  
  # Final aggregation
  aggregate_results:
    run: |
      echo "Aggregating results from all models..."
      sleep 1
      echo "Results aggregation complete!"
    type: shell
    depends_on: [train_model_x, train_model_y, train_model_z]
  
  generate_report:
    run: |
      echo "Generating final report..."
      sleep 1
      echo "Report generation complete!"
    type: shell
    depends_on: [aggregate_results]

settings:
  max_parallel_tasks: 4
  retries: 1
  log_dir: logs/
```

---

## 🔐 Checkpointing

LightFlow provides robust checkpointing capabilities:

- **Automatic Checkpointing**: Each task's completion status is stored in `.lightflow-checkpoint.json` files
- **Resume Capability**: On rerun, completed tasks are automatically skipped
- **Failure Recovery**: Failed tasks can be retried from the last successful checkpoint
- **Checkpoint Management**: List, clear, and manage checkpoints via CLI

### Checkpoint File Structure

```json
{
  "workflow_name": "my_workflow",
  "timestamp": "2025-07-30T15:30:05.842841",
  "completed_tasks": {
    "task1": {
      "success": true,
      "output": "Task completed successfully",
      "duration": 1.5,
      "exit_code": 0
    }
  },
  "total_tasks": 1,
  "version": "1.0"
}
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=lightflow --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_executor.py -v
```

### Test Coverage

- **48 test cases** covering all core functionality
- **44% code coverage** with comprehensive testing
- All tests passing ✅
- Example workflows tested and working ✅

---

## 📊 Performance

LightFlow is designed for small to medium-scale workflows:

- **Recommended**: Up to 50 tasks per workflow
- **Maximum**: 100+ tasks (with proper resource management)
- **Execution modes**: Thread, Process, Async
- **Parallelism**: Configurable worker pools (default: 4 workers)

### Performance Tips

1. **Use appropriate execution mode**:
   - `thread`: Good for I/O-bound tasks
   - `process`: Better for CPU-bound tasks
   - `async`: Best for I/O-bound tasks with high concurrency

2. **Optimize worker count**:
   - Set `max_parallel_tasks` based on your system resources
   - Monitor CPU and memory usage during execution

3. **Use checkpointing**:
   - Enable checkpointing for long-running workflows
   - Resume from checkpoints to avoid re-running completed tasks

---

## 🔧 Configuration

### Task Types

Currently supported task types:

- **shell**: Execute shell commands
- **python**: Execute Python code

### Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_parallel_tasks` | int | 4 | Maximum concurrent tasks |
| `retries` | int | 0 | Number of retry attempts |
| `log_dir` | string | "logs/" | Directory for log files |

### Task Configuration

```yaml
task_name:
  run: "command or code to execute"
  type: "shell|python"
  depends_on: ["task1", "task2"]
  cwd: "/optional/working/directory"
  env:
    PATH: "/custom/path"
  timeout: 30  # seconds
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Graphviz not available**
   ```bash
   pip install graphviz
   ```

2. **Permission errors**
   ```bash
   # Ensure write permissions for log and checkpoint directories
   chmod 755 logs/
   chmod 755 .lightflow-checkpoints/
   ```

3. **Task failures**
   ```bash
   # Check task output in log files
   ls logs/
   cat logs/task_taskname_timestamp.log
   ```

### Debug Mode

Enable debug logging:

```bash
python -m lightflow.cli.main run workflow.yaml --debug
```

### Getting Help

```bash
# Show help for all commands
python -m lightflow.cli.main --help

# Show help for specific command
python -m lightflow.cli.main run --help
```

---

## 📈 Future Enhancements

- [ ] Web dashboard for live task monitoring
- [ ] Cron-based scheduling
- [ ] Docker container support for task isolation
- [ ] GraphQL API to control tasks remotely
- [ ] Retry strategies per task (exponential backoff)
- [ ] More task plugin types (SQL, HTTP, Lambda)
- [ ] DAG visual enhancements
- [ ] Async I/O support

---

## 📜 License

MIT License — built for devs, data scientists, and tinkerers.

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd LightFlow

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### Areas for Contribution

- **More task plugin types** (e.g., SQL, HTTP, Lambda)
- **DAG visual enhancements**
- **Async I/O support**
- **Performance optimizations**
- **Documentation improvements**
- **Test coverage expansion**

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

---

## 🔗 Related Projects

- [Apache Airflow](https://airflow.apache.org/) - Full-featured workflow orchestration
- [Luigi](https://github.com/spotify/luigi) - Spotify's workflow engine
- [Prefect](https://www.prefect.io/) - Modern workflow orchestration
- [Dagster](https://dagster.io/) - Data orchestration platform

---

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the examples and tests

---

## 🙏 Acknowledgments

- Inspired by Apache Airflow
- Built with Python's concurrent.futures
- Visualization powered by Graphviz
- Rich CLI experience with Click

---

## 📝 Changelog

### v0.1.0
- Initial release
- Basic workflow execution
- YAML/JSON support
- DAG visualization
- Checkpointing system
- CLI interface
- Comprehensive test suite

---

## 🎯 Use Cases

LightFlow is perfect for:

- **Data Processing Pipelines**: ETL workflows, data transformation
- **Machine Learning**: Model training, evaluation, and deployment
- **CI/CD**: Build, test, and deployment automation
- **Reporting**: Automated report generation and distribution
- **Backup & Maintenance**: System maintenance and backup tasks
- **Research**: Experimental workflow automation

---

## 🚀 Getting Started Examples

### Example 1: Simple Data Pipeline

```yaml
workflow_name: data_pipeline
tasks:
  download:
    run: wget https://example.com/data.csv
    type: shell
    depends_on: []
  
  clean:
    run: python clean_data.py
    type: shell
    depends_on: [download]
  
  analyze:
    run: python analyze.py
    type: shell
    depends_on: [clean]
  
  report:
    run: python generate_report.py
    type: shell
    depends_on: [analyze]

settings:
  max_parallel_tasks: 2
```

### Example 2: ML Model Training

```yaml
workflow_name: ml_training
tasks:
  prepare_data:
    run: python prepare_data.py
    type: shell
    depends_on: []
  
  train_model:
    run: python train_model.py
    type: shell
    depends_on: [prepare_data]
  
  evaluate:
    run: python evaluate_model.py
    type: shell
    depends_on: [train_model]
  
  deploy:
    run: python deploy_model.py
    type: shell
    depends_on: [evaluate]

settings:
  max_parallel_tasks: 2
  retries: 3
```

---

**Ready to get started?** Check out the `examples/` directory for more sample workflows!

---

*Built with ❤️ for the Python community* 
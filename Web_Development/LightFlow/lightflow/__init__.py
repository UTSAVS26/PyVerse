"""
LightFlow: A Lightweight Parallel Task Pipeline Framework

A minimal, Pythonic alternative to Airflow — run dependent tasks in parallel 
using threads or processes, from a simple YAML workflow file.
"""

__version__ = "0.1.0"
__author__ = "Shivansh Katiyar"

from .engine.executor import WorkflowExecutor
from .engine.dag_builder import DAGBuilder
from .engine.checkpoint import CheckpointManager
from .engine.logger import Logger
from .parser.workflow_loader import WorkflowLoader

__all__ = [
    "WorkflowExecutor",
    "DAGBuilder", 
    "CheckpointManager",
    "Logger",
    "WorkflowLoader"
] 
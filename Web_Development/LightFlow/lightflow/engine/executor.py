"""
WorkflowExecutor - Handles parallel task execution using threads and processes.
"""

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .logger import Logger
from .checkpoint import CheckpointManager


class ExecutionMode(Enum):
    """Execution modes for tasks."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_name: str
    success: bool
    output: str
    error: Optional[str]
    duration: float
    exit_code: int


class WorkflowExecutor:
    """Executes workflow tasks in parallel with dependency management."""
    
    def __init__(self, max_workers: int = 4, mode: ExecutionMode = ExecutionMode.THREAD):
        self.max_workers = max_workers
        self.mode = mode
        self.logger = Logger()
        self.checkpoint_manager = CheckpointManager()
        self._running_tasks: Dict[str, Any] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._failed_tasks: Dict[str, TaskResult] = {}
        
    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, TaskResult]:
        """
        Execute a complete workflow with dependency management.
        
        Args:
            workflow: Workflow configuration dictionary
            
        Returns:
            Dictionary mapping task names to their results
        """
        tasks = workflow.get('tasks', {})
        settings = workflow.get('settings', {})
        
        # Load checkpoint if exists
        checkpoint = self.checkpoint_manager.load_checkpoint(workflow.get('workflow_name', 'default'))
        
        # Build execution order based on dependencies
        execution_order = self._build_execution_order(tasks)
        
        self.logger.info(f"Starting workflow execution with {len(tasks)} tasks")
        self.logger.info(f"Execution mode: {self.mode.value}")
        self.logger.info(f"Max parallel tasks: {self.max_workers}")
        
        # Execute tasks in order
        for task_batch in execution_order:
            self._execute_task_batch(task_batch, tasks, checkpoint)
            
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            workflow.get('workflow_name', 'default'),
            self._completed_tasks
        )
        
        return {**self._completed_tasks, **self._failed_tasks}
    
    def _build_execution_order(self, tasks: Dict[str, Any]) -> List[List[str]]:
        """
        Build execution order based on task dependencies.
        
        Args:
            tasks: Dictionary of task configurations
            
        Returns:
            List of task batches that can be executed in parallel
        """
        # Create dependency graph
        dependencies = {}
        for task_name, task_config in tasks.items():
            deps = task_config.get('depends_on', [])
            dependencies[task_name] = deps
            
        # Topological sort to find execution order
        execution_order = []
        completed = set()
        
        while len(completed) < len(tasks):
            batch = []
            for task_name in tasks:
                if task_name in completed:
                    continue
                    
                # Check if all dependencies are completed
                deps = dependencies.get(task_name, [])
                if all(dep in completed for dep in deps):
                    batch.append(task_name)
                    
            if not batch:
                # Circular dependency detected
                remaining = set(tasks.keys()) - completed
                raise ValueError(f"Circular dependency detected in tasks: {remaining}")
                
            execution_order.append(batch)
            completed.update(batch)
            
        return execution_order
    
    def _execute_task_batch(self, task_batch: List[str], tasks: Dict[str, Any], 
                           checkpoint: Dict[str, Any]):
        """
        Execute a batch of tasks in parallel.
        
        Args:
            task_batch: List of task names to execute
            tasks: Task configurations
            checkpoint: Current checkpoint data
        """
        # Filter out already completed tasks
        pending_tasks = [task for task in task_batch 
                        if task not in checkpoint.get('completed', {})]
        
        if not pending_tasks:
            self.logger.info(f"All tasks in batch already completed: {task_batch}")
            return
            
        self.logger.info(f"Executing batch: {pending_tasks}")
        
        if self.mode == ExecutionMode.THREAD:
            self._execute_with_threads(pending_tasks, tasks)
        elif self.mode == ExecutionMode.PROCESS:
            self._execute_with_processes(pending_tasks, tasks)
        elif self.mode == ExecutionMode.ASYNC:
            asyncio.run(self._execute_with_async(pending_tasks, tasks))
    
    def _execute_with_threads(self, task_names: List[str], tasks: Dict[str, Any]):
        """Execute tasks using ThreadPoolExecutor."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, name, tasks[name]): name
                for name in task_names
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    self._completed_tasks[task_name] = result
                    self.logger.info(f"Task {task_name} completed successfully")
                except Exception as e:
                    error_result = TaskResult(
                        task_name=task_name,
                        success=False,
                        output="",
                        error=str(e),
                        duration=0.0,
                        exit_code=1
                    )
                    self._failed_tasks[task_name] = error_result
                    self.logger.error(f"Task {task_name} failed: {e}")
    
    def _execute_with_processes(self, task_names: List[str], tasks: Dict[str, Any]):
        """Execute tasks using ProcessPoolExecutor."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, name, tasks[name]): name
                for name in task_names
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    self._completed_tasks[task_name] = result
                    self.logger.info(f"Task {task_name} completed successfully")
                except Exception as e:
                    error_result = TaskResult(
                        task_name=task_name,
                        success=False,
                        output="",
                        error=str(e),
                        duration=0.0,
                        exit_code=1
                    )
                    self._failed_tasks[task_name] = error_result
                    self.logger.error(f"Task {task_name} failed: {e}")
    
    async def _execute_with_async(self, task_names: List[str], tasks: Dict[str, Any]):
        """Execute tasks using asyncio."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def execute_with_semaphore(task_name: str, task_config: Dict[str, Any]):
            async with semaphore:
                return await self._execute_single_task_async(task_name, task_config)
        
        tasks_to_execute = [
            execute_with_semaphore(name, tasks[name]) 
            for name in task_names
        ]
        
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        
        for i, result in enumerate(results):
            task_name = task_names[i]
            if isinstance(result, Exception):
                error_result = TaskResult(
                    task_name=task_name,
                    success=False,
                    output="",
                    error=str(result),
                    duration=0.0,
                    exit_code=1
                )
                self._failed_tasks[task_name] = error_result
                self.logger.error(f"Task {task_name} failed: {result}")
            else:
                self._completed_tasks[task_name] = result
                self.logger.info(f"Task {task_name} completed successfully")
    
    def _execute_single_task(self, task_name: str, task_config: Dict[str, Any]) -> TaskResult:
        """
        Execute a single task.
        
        Args:
            task_name: Name of the task
            task_config: Task configuration
            
        Returns:
            TaskResult with execution details
        """
        start_time = time.time()
        
        try:
            # Get task type and execute accordingly
            task_type = task_config.get('type', 'shell')
            
            if task_type == 'shell':
                output, exit_code = self._execute_shell_task(task_config)
            elif task_type == 'python':
                output, exit_code = self._execute_python_task(task_config)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            duration = time.time() - start_time
            
            return TaskResult(
                task_name=task_name,
                success=exit_code == 0,
                output=output,
                error=None,
                duration=duration,
                exit_code=exit_code
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TaskResult(
                task_name=task_name,
                success=False,
                output="",
                error=str(e),
                duration=duration,
                exit_code=1
            )
    
    async def _execute_single_task_async(self, task_name: str, task_config: Dict[str, Any]) -> TaskResult:
        """Async version of single task execution."""
        start_time = time.time()
        
        try:
            task_type = task_config.get('type', 'shell')
            
            if task_type == 'shell':
                output, exit_code = await self._execute_shell_task_async(task_config)
            elif task_type == 'python':
                output, exit_code = await self._execute_python_task_async(task_config)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            duration = time.time() - start_time
            
            return TaskResult(
                task_name=task_name,
                success=exit_code == 0,
                output=output,
                error=None,
                duration=duration,
                exit_code=exit_code
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TaskResult(
                task_name=task_name,
                success=False,
                output="",
                error=str(e),
                duration=duration,
                exit_code=1
            )
    
    def _execute_shell_task(self, task_config: Dict[str, Any]) -> tuple[str, int]:
        """Execute a shell command task."""
        import subprocess
        
        command = task_config['run']
        cwd = task_config.get('cwd', None)
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        output = stdout
        if stderr:
            output += f"\nSTDERR:\n{stderr}"
            
        return output, exit_code
    
    async def _execute_shell_task_async(self, task_config: Dict[str, Any]) -> tuple[str, int]:
        """Execute a shell command task asynchronously."""
        import subprocess
        
        command = task_config['run']
        cwd = task_config.get('cwd', None)
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        stdout, stderr = await process.communicate()
        exit_code = process.returncode
        
        output = stdout.decode() if stdout else ""
        if stderr:
            output += f"\nSTDERR:\n{stderr.decode()}"
            
        return output, exit_code
    
    def _execute_python_task(self, task_config: Dict[str, Any]) -> tuple[str, int]:
        """Execute a Python function task."""
        import importlib.util
        import sys
        from io import StringIO
        
        # Capture stdout
        old_stdout = sys.stdout
        output_buffer = StringIO()
        sys.stdout = output_buffer
        
        try:
            # Execute Python code
            code = task_config['run']
            exec(code, {})
            
            output = output_buffer.getvalue()
            return output, 0
            
        except Exception as e:
            output = output_buffer.getvalue() + f"\nError: {str(e)}"
            return output, 1
        finally:
            sys.stdout = old_stdout
    
    async def _execute_python_task_async(self, task_config: Dict[str, Any]) -> tuple[str, int]:
        """Execute a Python function task asynchronously."""
        # For now, just call the sync version
        return self._execute_python_task(task_config) 
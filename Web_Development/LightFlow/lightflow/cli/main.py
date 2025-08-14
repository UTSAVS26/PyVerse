"""
LightFlow CLI - Command line interface for LightFlow framework.
"""

import click
import click
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightflow.engine.executor import WorkflowExecutor, ExecutionMode
from lightflow.engine.dag_builder import DAGBuilder
from lightflow.engine.checkpoint import CheckpointManager
from lightflow.parser.workflow_loader import WorkflowLoader


@click.group()
@click.version_option(version="0.1.0")
def main():
    """LightFlow - A Lightweight Parallel Task Pipeline Framework"""
    pass


@main.command()
@click.argument('workflow_file', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['thread', 'process', 'async']), 
              default='thread', help='Execution mode')
@click.option('--max-workers', type=int, default=4, 
              help='Maximum number of parallel workers')
@click.option('--resume', is_flag=True, 
              help='Resume from checkpoint if available')
@click.option('--dry-run', is_flag=True, 
              help='Show execution plan without running')
def run(workflow_file, mode, max_workers, resume, dry_run):
    """Run a LightFlow workflow."""
    try:
        # Load workflow
        loader = WorkflowLoader()
        workflow = loader.load_workflow(workflow_file)
        
        # Validate workflow
        is_valid, errors = loader.validate_workflow(workflow)
        if not is_valid:
            click.echo("Workflow validation failed:")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)
        
        # Build DAG
        dag_builder = DAGBuilder()
        dag = dag_builder.build_dag(workflow)
        
        # Validate DAG
        dag_valid, dag_errors = dag_builder.validate_dag(dag)
        if not dag_valid:
            click.echo("DAG validation failed:")
            for error in dag_errors:
                click.echo(f"  - {error}")
            sys.exit(1)
        
        # Show workflow info
        workflow_info = loader.get_workflow_info(workflow)
        click.echo(f"Workflow: {workflow_info['workflow_name']}")
        click.echo(f"Total tasks: {workflow_info['total_tasks']}")
        click.echo(f"Task types: {workflow_info['task_types']}")
        click.echo(f"Entry tasks: {workflow_info['entry_tasks']}")
        click.echo(f"Exit tasks: {workflow_info['exit_tasks']}")
        
        # Show execution plan
        execution_order = dag_builder.get_execution_order(dag)
        click.echo(f"\nExecution plan ({len(execution_order)} batches):")
        for i, batch in enumerate(execution_order):
            click.echo(f"  Batch {i+1}: {', '.join(batch)}")
        
        if dry_run:
            click.echo("\nDry run completed. Use --resume to run the workflow.")
            return
        
        # Execute workflow
        executor = WorkflowExecutor(
            max_workers=max_workers,
            mode=ExecutionMode(mode)
        )
        
        results = executor.execute_workflow(workflow)
        
        # Show results
        click.echo("\nExecution Results:")
        for task_name, result in results.items():
            status = "✓ Success" if result.success else "✗ Failed"
            click.echo(f"  {task_name}: {status} ({result.duration:.2f}s)")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('workflow_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for DAG visualization')
@click.option('--format', type=click.Choice(['svg', 'png', 'pdf']), 
              default='svg', help='Output format')
def dag(workflow_file, output, format):
    """Generate DAG visualization for a workflow."""
    try:
        # Load workflow
        loader = WorkflowLoader()
        workflow = loader.load_workflow(workflow_file)
        
        # Build DAG
        dag_builder = DAGBuilder()
        dag = dag_builder.build_dag(workflow)
        
        # Generate visualization
        if output:
            output_file = f"{output}.{format}"
            dag_builder.visualize_dag(dag, output_file)
        else:
            dag_builder.visualize_dag(dag)
        click.echo("DAG visualization generated successfully!")
        if output:
            click.echo(f"Output saved to: {output_file}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('workflow_name')
@click.option('--task', '-t', help='Show logs for specific task')
@click.option('--checkpoint', '-c', help='Show checkpoint information')
def logs(workflow_name, task, checkpoint):
    """Show workflow logs and checkpoint information."""
    try:
        checkpoint_manager = CheckpointManager()
        
        if checkpoint:
            # Show specific checkpoint info
            checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint)
            click.echo("Checkpoint Information:")
            for key, value in checkpoint_info.items():
                click.echo(f"  {key}: {value}")
        else:
         else:
             # Show latest checkpoint
             if task:
                 # Show specific task result
                 task_result = checkpoint_manager.get_task_result(workflow_name, task)
                 if task_result:
                     click.echo(f"Task: {task}")
                     click.echo(f"Success: {task_result.get('success', False)}")
                     click.echo(f"Duration: {task_result.get('duration', 0):.2f}s")
                     click.echo(f"Exit Code: {task_result.get('exit_code', 0)}")
                     if task_result.get('output'):
                         click.echo(f"Output:\n{task_result['output']}")
                 else:
                     click.echo(f"Task '{task}' not found in checkpoint")
             else:
                # Show all completed tasks
                completed_tasks = checkpoint_manager.get_completed_tasks(workflow_name)
                if completed_tasks:
                    click.echo(f"Completed tasks for workflow '{workflow_name}':")
                    for task_name in completed_tasks:
                        click.echo(f"  - {task_name}")
                else:
                    click.echo(f"No completed tasks found for workflow '{workflow_name}'")
                    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('workflow_file', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['thread', 'process', 'async']),
              default='thread', help='Execution mode')
@click.option('--max-workers', type=int, default=4,
              help='Maximum number of parallel workers')
def resume(workflow_file, mode, max_workers):
    """Resume a workflow from checkpoint."""
    try:
        # Load workflow
        loader = WorkflowLoader()
        workflow = loader.load_workflow(workflow_file)
        
        # Check for checkpoint
        checkpoint_manager = CheckpointManager()
        workflow_name = workflow.get('workflow_name', 'default')
        checkpoint_data = checkpoint_manager.load_checkpoint(workflow_name)
        
        completed_tasks = list(checkpoint_data.get('completed_tasks', {}).keys())
        if completed_tasks:
            click.echo(f"Found checkpoint with {len(completed_tasks)} completed tasks:")
            for task in completed_tasks:
                click.echo(f"  - {task}")
            click.echo("\nResuming workflow...")
        else:
            click.echo("No checkpoint found. Running workflow from start.")
        
        ctx = click.get_current_context()
        ctx.invoke(run, workflow_file=workflow_file, mode=mode,
                   max_workers=max_workers, dry_run=False)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('workflow_name')
@click.option('--all', is_flag=True, help='Clear all checkpoints')
def clear(workflow_name, all):
    """Clear checkpoints for a workflow."""
    try:
        checkpoint_manager = CheckpointManager()

        if all:
            deleted_count = checkpoint_manager.clear_checkpoints()
            click.echo(f"Cleared {deleted_count} checkpoints")
        else:
            deleted_count = checkpoint_manager.clear_checkpoints(workflow_name)
            click.echo(f"Cleared {deleted_count} checkpoints for workflow '{workflow_name}'")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('workflow_name')
def list_checkpoints(workflow_name):
    """List available checkpoints for a workflow."""
    try:
        checkpoint_manager = CheckpointManager()
        checkpoints = checkpoint_manager.list_checkpoints(workflow_name)
        
        if checkpoints:
            click.echo(f"Checkpoints for workflow '{workflow_name}':")
            for checkpoint in checkpoints:
                info = checkpoint_manager.get_checkpoint_info(checkpoint)
                click.echo(f"  {checkpoint}")
                click.echo(f"    Timestamp: {info.get('timestamp', 'Unknown')}")
                click.echo(f"    Tasks: {info.get('total_tasks', 0)}")
        else:
            click.echo(f"No checkpoints found for workflow '{workflow_name}'")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('workflow_name')
def template(workflow_name):
    """Create a workflow template."""
    try:
        loader = WorkflowLoader()
        template = loader.create_workflow_template(workflow_name)
        
        # Save template
        template_file = f"{workflow_name}_template.yaml"
        loader.save_workflow(template, template_file)
        
        click.echo(f"Workflow template created: {template_file}")
        click.echo("Edit the template and run with: lightflow run <filename>")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('workflow_file', type=click.Path(exists=True))
def validate(workflow_file):
    """Validate a workflow file."""
    try:
        # Load workflow
        loader = WorkflowLoader()
        workflow = loader.load_workflow(workflow_file)
        
        # Validate workflow
        is_valid, errors = loader.validate_workflow(workflow)
        
        if is_valid:
            click.echo("✓ Workflow is valid!")
            
            # Show workflow info
            workflow_info = loader.get_workflow_info(workflow)
            click.echo(f"Workflow: {workflow_info['workflow_name']}")
            click.echo(f"Total tasks: {workflow_info['total_tasks']}")
            click.echo(f"Task types: {workflow_info['task_types']}")
            
            # Build and validate DAG
            dag_builder = DAGBuilder()
            dag = dag_builder.build_dag(workflow)
            dag_valid, dag_errors = dag_builder.validate_dag(dag)
            
            if dag_valid:
                click.echo("✓ DAG is valid!")
                execution_order = dag_builder.get_execution_order(dag)
                click.echo(f"Execution batches: {len(execution_order)}")
            else:
                click.echo("✗ DAG validation failed:")
                for error in dag_errors:
                    click.echo(f"  - {error}")
                    
        else:
            click.echo("✗ Workflow validation failed:")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main() 
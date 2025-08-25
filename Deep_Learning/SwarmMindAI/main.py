#!/usr/bin/env python3
"""
SwarmMindAI - Advanced Multi-Agent Swarm Intelligence Framework

Main entry point for running swarm simulations.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.environment import SwarmEnvironment
from src.visualization import SwarmVisualizer
from src.algorithms import MultiAgentPPO, MultiAgentDQN


def main():
    """Main entry point for SwarmMindAI."""
    parser = argparse.ArgumentParser(
        description="SwarmMindAI - Advanced Multi-Agent Swarm Intelligence Framework"
    )
    
    parser.add_argument(
        "--world-size", 
        type=int, 
        nargs=2, 
        default=[1000, 1000],
        help="World dimensions (width height)"
    )
    
    parser.add_argument(
        "--num-agents", 
        type=int, 
        default=20,
        help="Number of agents in the swarm"
    )
    
    parser.add_argument(
        "--agent-types", 
        nargs="+", 
        default=["explorer", "collector", "coordinator"],
        help="Types of agents to create"
    )
    
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=1000,
        help="Maximum simulation steps"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Enable real-time visualization"
    )
    
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run simulation without visualization"
    )
    
    parser.add_argument(
        "--save-results", 
        action="store_true",
        help="Save simulation results to file"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    print("🐝 SwarmMindAI - Advanced Multi-Agent Swarm Intelligence Framework")
    print("=" * 70)
    
    try:
        # Initialize environment
        print(f"Initializing environment: {args.world_size[0]}x{args.world_size[1]}")
        print(f"Creating swarm with {args.num_agents} agents: {', '.join(args.agent_types)}")
        
        env = SwarmEnvironment(
            world_size=tuple(args.world_size),
            num_agents=args.num_agents,
            agent_types=args.agent_types,
            seed=args.seed,
            max_steps=args.max_steps
        )
        
        # Initialize visualization if requested
        visualizer = None
        if args.visualize and not args.headless:
            try:
                visualizer = SwarmVisualizer(env)
                print("✅ Visualization initialized")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
                print("Continuing without visualization...")
        
        # Run simulation
        print("\n🚀 Starting simulation...")
        env.start_simulation()
        
        step_count = 0
        start_time = time.time()
        
        try:
            while step_count < args.max_steps:
                # Execute simulation step
                step_result = env.step()
                step_count += 1
                
                # Update visualization
                if visualizer and not args.headless:
                    visualizer.update(step_result)
                
                # Print progress
                if step_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_second = step_count / elapsed_time
                    print(f"Step {step_count}/{args.max_steps} "
                          f"({steps_per_second:.1f} steps/sec)")
                    
                    # Print metrics
                    metrics = env.get_environment_state()
                    print(f"  Active agents: {metrics['swarm_metrics']['active_agents']}")
                    print(f"  Task completion: {metrics['task_statistics']['completion_rate']:.2f}")
                    print(f"  Coordination score: {metrics['swarm_metrics']['coordination_score']:.2f}")
                
                # Check if simulation is done
                if step_result.get("done", False):
                    print(f"\n✅ Simulation completed after {step_count} steps")
                    break
                
                # Small delay for visualization
                if visualizer and not args.headless:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
        
        finally:
            # Final statistics
            final_metrics = env.get_environment_state()
            total_time = time.time() - start_time
            
            print("\n📊 Simulation Results:")
            print("=" * 40)
            print(f"Total steps: {step_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Steps per second: {step_count/total_time:.1f}")
            print(f"Final coordination score: {final_metrics['swarm_metrics']['coordination_score']:.3f}")
            print(f"Task completion rate: {final_metrics['task_statistics']['completion_rate']:.3f}")
            print(f"Active agents: {final_metrics['swarm_metrics']['active_agents']}")
            
            # Save results if requested
            if args.save_results:
                save_simulation_results(env, step_count, total_time)
            
            # Cleanup
            if visualizer:
                visualizer.close()
            env.close()
            
            print("\n🎉 Simulation finished successfully!")
    
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def save_simulation_results(env, step_count, total_time):
    """Save simulation results to file."""
    try:
        from datetime import datetime
        import json
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"swarm_simulation_{timestamp}.json"
        
        # Collect results
        results = {
            "timestamp": timestamp,
            "simulation_stats": {
                "total_steps": step_count,
                "total_time": total_time,
                "steps_per_second": step_count / total_time
            },
            "environment_state": env.get_environment_state(),
            "swarm_metrics": env.swarm.get_swarm_state()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
    
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")


if __name__ == "__main__":
    sys.exit(main())

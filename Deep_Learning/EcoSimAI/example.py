#!/usr/bin/env python3
"""
Example script demonstrating EcoSimAI functionality.
Shows different simulation scenarios and configurations.
"""

import time
from environment import Environment, CellType
from agents import Prey, Predator, Plant
from rl_agent import RLPrey, RLPredator
from simulate import run_headless_simulation, save_simulation_stats

def example_basic_simulation():
    """Example 1: Basic ecosystem simulation."""
    print("🌱 Example 1: Basic Ecosystem Simulation")
    print("="*50)
    
    # Create environment
    env = Environment(width=30, height=30, plant_growth_rate=0.15)
    
    print(f"Environment: {env.width}x{env.height} grid")
    print(f"Initial plants: {env.get_statistics()['plants']}")
    print(f"Initial prey: {env.get_statistics()['prey']}")
    print(f"Initial predators: {env.get_statistics()['predators']}")
    
    # Run simulation
    stats = run_headless_simulation(env, max_steps=100, print_interval=20)
    
    print(f"\nFinal Statistics:")
    final_stats = stats[-1]
    print(f"Final plants: {final_stats['plants']}")
    print(f"Final prey: {final_stats['prey']}")
    print(f"Final predators: {final_stats['predators']}")
    print(f"Total steps: {len(stats)}")

def example_rl_agents():
    """Example 2: Simulation with RL agents."""
    print("\n🧠 Example 2: RL Agents Simulation")
    print("="*50)
    
    # Create environment with RL agents
    env = Environment(width=20, height=20)
    
    # Clear existing agents and add RL agents
    env.agents.clear()
    env.agent_positions.clear()
    env.grid.fill(CellType.EMPTY.value)
    
    # Add plants
    for _ in range(30):
        pos = env._get_random_empty_position()
        if pos:
            env.grid[pos.y, pos.x] = CellType.PLANT.value
    
    # Add RL prey
    for i in range(10):
        pos = env._get_random_empty_position()
        if pos:
            rl_prey = RLPrey(env._get_next_agent_id(), pos, energy=50)
            env._add_agent(rl_prey, pos)
    
    # Add RL predators
    for i in range(5):
        pos = env._get_random_empty_position()
        if pos:
            rl_predator = RLPredator(env._get_next_agent_id(), pos, energy=50)
            env._add_agent(rl_predator, pos)
    
    print(f"Environment: {env.width}x{env.height} grid")
    print(f"RL Prey: {env.get_statistics()['prey']}")
    print(f"RL Predators: {env.get_statistics()['predators']}")
    
    # Run simulation
    stats = run_headless_simulation(env, max_steps=150, print_interval=30)
    
    # Check RL agent learning
    rl_agents = [agent for agent in env.agents.values() 
                if isinstance(agent, (RLPrey, RLPredator))]
    
    if rl_agents:
        avg_memory = sum(len(agent.memory) for agent in rl_agents) / len(rl_agents)
        print(f"\nRL Agent Learning:")
        print(f"Average memory size: {avg_memory:.1f} experiences")
        print(f"Total RL agents: {len(rl_agents)}")

def example_ecosystem_scenarios():
    """Example 3: Different ecosystem scenarios."""
    print("\n🌍 Example 3: Ecosystem Scenarios")
    print("="*50)
    
    scenarios = [
        ("High Plant Growth", {"plant_growth_rate": 0.3, "initial_plants": 80}),
        ("Low Plant Growth", {"plant_growth_rate": 0.05, "initial_plants": 20}),
        ("Many Predators", {"initial_predators": 40, "initial_prey": 30}),
        ("Many Prey", {"initial_prey": 80, "initial_predators": 10}),
    ]
    
    for scenario_name, params in scenarios:
        print(f"\n📊 Scenario: {scenario_name}")
        print("-" * 30)
        
        # Create environment with scenario parameters
        env = Environment(width=25, height=25, **params)
        
        # Run short simulation
        stats = run_headless_simulation(env, max_steps=50, print_interval=25)
        
        # Analyze results
        final_stats = stats[-1]
        print(f"Final populations:")
        print(f"  Plants: {final_stats['plants']}")
        print(f"  Prey: {final_stats['prey']}")
        print(f"  Predators: {final_stats['predators']}")
        
        # Determine outcome
        if final_stats['prey'] == 0 and final_stats['predators'] == 0:
            outcome = "Extinction"
        elif final_stats['prey'] == 0:
            outcome = "Prey Extinction"
        elif final_stats['predators'] == 0:
            outcome = "Predator Extinction"
        else:
            outcome = "Balanced"
        
        print(f"Outcome: {outcome}")

def example_agent_behavior():
    """Example 4: Demonstrate agent behavior patterns."""
    print("\n🎭 Example 4: Agent Behavior Patterns")
    print("="*50)
    
    # Create small environment for detailed observation
    env = Environment(width=15, height=15)
    
    # Add specific agents at known positions
    prey = Prey(0, env._get_random_empty_position(), energy=60)
    predator = Predator(1, env._get_random_empty_position(), energy=70)
    env._add_agent(prey, env._get_random_empty_position())
    env._add_agent(predator, env._get_random_empty_position())
    
    print("Tracking agent behavior over 10 steps:")
    print("Step | Prey Energy | Predator Energy | Distance")
    print("-" * 45)
    
    for step in range(10):
        env.step()
        stats = env.get_statistics()
        
        # Calculate distance between agents
        prey_pos = env.agent_positions.get(prey.id)
        predator_pos = env.agent_positions.get(predator.id)
        
        if prey_pos and predator_pos:
            distance = prey_pos.distance_to(predator_pos)
        else:
            distance = "N/A (agent died)"
        
        print(f"{step:4d} | {prey.energy:11.1f} | {predator.energy:14.1f} | {distance}")

def example_data_export():
    """Example 5: Data export and analysis."""
    print("\n📊 Example 5: Data Export and Analysis")
    print("="*50)
    
    # Create environment
    env = Environment(width=20, height=20)
    
    # Run simulation and collect data
    stats = run_headless_simulation(env, max_steps=80, print_interval=20)
    
    # Save data to CSV
    filename = "ecosystem_data_example.csv"
    save_simulation_stats(stats, filename)
    
    print(f"Data saved to: {filename}")
    print(f"Total data points: {len(stats)}")
    
    # Basic analysis
    if len(stats) > 1:
        plants = [s['plants'] for s in stats]
        prey = [s['prey'] for s in stats]
        predators = [s['predators'] for s in stats]
        
        print(f"\nPopulation Analysis:")
        print(f"Plants: min={min(plants)}, max={max(plants)}, avg={sum(plants)/len(plants):.1f}")
        print(f"Prey: min={min(prey)}, max={max(prey)}, avg={sum(prey)/len(prey):.1f}")
        print(f"Predators: min={min(predators)}, max={max(predators)}, avg={sum(predators)/len(predators):.1f}")

def main():
    """Run all examples."""
    print("🌱 EcoSimAI - Example Demonstrations")
    print("="*60)
    print("This script demonstrates various features of the EcoSimAI project.")
    print("Each example shows different aspects of the ecosystem simulation.\n")
    
    try:
        # Run examples
        example_basic_simulation()
        time.sleep(1)
        
        example_rl_agents()
        time.sleep(1)
        
        example_ecosystem_scenarios()
        time.sleep(1)
        
        example_agent_behavior()
        time.sleep(1)
        
        example_data_export()
        
        print(f"\n🎉 All examples completed successfully!")
        print(f"Check the generated CSV file for detailed data analysis.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

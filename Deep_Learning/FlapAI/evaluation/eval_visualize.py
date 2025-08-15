import sys
import os
import time
from typing import Dict, Any, Optional
import pygame

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBirdGame
from agents.neat_agent import NEATAgent
from agents.dqn_agent import DQNAgent
from agents.base_agent import RandomAgent, HumanAgent

class AgentVisualizer:
    """Visualize trained agents playing Flappy Bird."""
    
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.game = FlappyBirdGame(fps=fps)
        
    def visualize_agent(self, agent_type: str, model_path: str = None, 
                       episodes: int = 5, delay: float = 0.1) -> None:
        """
        Visualize an agent playing the game.
        
        Args:
            agent_type: Type of agent ('neat', 'dqn', 'random', 'human')
            model_path: Path to saved model (for NEAT/DQN)
            episodes: Number of episodes to run
            delay: Delay between frames (for slower viewing)
        """
        # Create agent
        agent = self._create_agent(agent_type, model_path)
        
        print(f"Visualizing {agent_type.upper()} agent...")
        print(f"Episodes: {episodes}")
        print(f"Model path: {model_path if model_path else 'None'}")
        print("-" * 50)
        
        total_score = 0
        best_score = 0
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            # Reset game
            state = self.game.reset()
            agent.reset_episode()
            
            episode_score = 0
            frame_count = 0
            done = False
            
            while not done:
                # Handle pygame events
                if not self.game.handle_events():
                    print("Game closed by user")
                    return
                    
                # Get action from agent
                action = agent.get_action(state)
                
                # Take step in game
                next_state, reward, done, info = self.game.step(action)
                
                # Update agent
                agent.update(state, action, reward, next_state, done)
                
                # Update statistics
                episode_score = info.get('score', 0)
                frame_count += 1
                
                # Render game
                self.game.render()
                
                # Add delay for slower viewing
                if delay > 0:
                    time.sleep(delay)
                    
                state = next_state
                
            # Episode complete
            total_score += episode_score
            if episode_score > best_score:
                best_score = episode_score
                
            print(f"  Score: {episode_score}")
            print(f"  Frames: {frame_count}")
            
        # Final statistics
        avg_score = total_score / episodes
        print(f"\nFinal Statistics:")
        print(f"  Average Score: {avg_score:.1f}")
        print(f"  Best Score: {best_score}")
        print(f"  Total Episodes: {episodes}")
        
    def _create_agent(self, agent_type: str, model_path: str = None):
        """Create an agent of the specified type."""
        if agent_type.lower() == 'neat':
            agent = NEATAgent()
            if model_path and os.path.exists(model_path):
                agent.load(model_path)
                print(f"Loaded NEAT agent from {model_path}")
            return agent
            
        elif agent_type.lower() == 'dqn':
            agent = DQNAgent()
            if model_path and os.path.exists(model_path):
                agent.load(model_path)
                print(f"Loaded DQN agent from {model_path}")
            return agent
            
        elif agent_type.lower() == 'random':
            return RandomAgent()
            
        elif agent_type.lower() == 'human':
            return HumanAgent()
            
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    def compare_agents(self, agents: Dict[str, str], episodes: int = 3) -> Dict[str, float]:
        """
        Compare multiple agents.
        
        Args:
            agents: Dictionary mapping agent names to (type, model_path)
            episodes: Number of episodes per agent
            
        Returns:
            Dictionary mapping agent names to average scores
        """
        results = {}
        
        print(f"Comparing {len(agents)} agents over {episodes} episodes each...")
        print("-" * 50)
        
        for agent_name, (agent_type, model_path) in agents.items():
            print(f"\nTesting {agent_name} ({agent_type})...")
            
            agent = self._create_agent(agent_type, model_path)
            total_score = 0
            
            for episode in range(episodes):
                # Reset game
                state = self.game.reset()
                agent.reset_episode()
                
                done = False
                while not done:
                    # Handle events
                    if not self.game.handle_events():
                        return results
                        
                    # Get action and step
                    action = agent.get_action(state)
                    next_state, reward, done, info = self.game.step(action)
                    agent.update(state, action, reward, next_state, done)
                    
                    # Render
                    self.game.render()
                    
                    state = next_state
                    
                # Add episode score
                episode_score = info.get('score', 0)
                total_score += episode_score
                
            # Calculate average score
            avg_score = total_score / episodes
            results[agent_name] = avg_score
            
            print(f"  Average Score: {avg_score:.1f}")
            
        # Print comparison results
        print(f"\nComparison Results:")
        print("-" * 30)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (agent_name, score) in enumerate(sorted_results):
            print(f"{i+1}. {agent_name}: {score:.1f}")
            
        return results

def visualize_agent(agent_type: str, model_path: str = None, 
                   episodes: int = 5, delay: float = 0.1) -> None:
    """
    Convenience function to visualize an agent.
    
    Args:
        agent_type: Type of agent ('neat', 'dqn', 'random', 'human')
        model_path: Path to saved model
        episodes: Number of episodes to run
        delay: Delay between frames
    """
    visualizer = AgentVisualizer()
    visualizer.visualize_agent(agent_type, model_path, episodes, delay)

def main():
    """Main function for agent visualization."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trained agents for FlapAI')
    parser.add_argument('--agent-type', type=str, required=True,
                       choices=['neat', 'dqn', 'random', 'human'],
                       help='Type of agent to visualize')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between frames (seconds)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple agents')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple agents
        agents = {
            'Random': ('random', None),
            'NEAT': ('neat', 'models/best_neat_gen_final.pkl'),
            'DQN': ('dqn', 'models/best_dqn_ep_final.pth')
        }
        
        # Only include agents with existing models
        available_agents = {}
        for name, (agent_type, model_path) in agents.items():
            if model_path is None or os.path.exists(model_path):
                available_agents[name] = (agent_type, model_path)
                
        if not available_agents:
            print("No trained models found. Training a quick NEAT agent...")
            from training.train_neat import train_neat
            train_neat(generations=10, render=False)
            available_agents['NEAT'] = ('neat', 'models/best_neat_gen_final.pkl')
            
        visualizer = AgentVisualizer()
        visualizer.compare_agents(available_agents, args.episodes)
        
    else:
        # Visualize single agent
        visualize_agent(args.agent_type, args.model_path, args.episodes, args.delay)

if __name__ == "__main__":
    main() 
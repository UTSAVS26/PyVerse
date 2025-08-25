"""
Real-time visualization for SwarmMindAI swarm simulation.
"""

import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional, Any
import time
from ..environment import SwarmEnvironment


class SwarmVisualizer:
    """
    Real-time visualization of the swarm simulation using Pygame.
    
    Features:
    - Real-time agent movement visualization
    - Resource and obstacle display
    - Performance metrics overlay
    - Interactive controls
    - Multiple view modes
    """
    
    def __init__(self, environment: SwarmEnvironment, 
                 window_size: Tuple[int, int] = (1200, 800),
                 fps: int = 60):
        """
        Initialize the swarm visualizer.
        
        Args:
            environment: SwarmEnvironment to visualize
            window_size: Size of the visualization window
            fps: Target frames per second
        """
        self.environment = environment
        self.window_size = window_size
        self.fps = fps
        
        # Pygame initialization
        pygame.init()
        pygame.font.init()  # Initialize font module
        
        # Try to initialize display, fall back to headless mode if needed
        try:
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("SwarmMindAI - Advanced Swarm Intelligence Simulation")
            self.headless = False
        except pygame.error:
            # Fall back to headless mode for testing environments
            self.screen = None
            self.headless = True
        
        # Colors
        self.colors = {
            "background": (20, 20, 40),
            "world_border": (60, 60, 80),
            "agent_explorer": (0, 255, 0),      # Green
            "agent_collector": (255, 165, 0),   # Orange
            "agent_coordinator": (0, 191, 255), # Deep Sky Blue
            "resource_food": (255, 255, 0),     # Yellow
            "resource_mineral": (128, 128, 128), # Gray
            "resource_energy": (255, 0, 255),   # Magenta
            "resource_water": (0, 255, 255),    # Cyan
            "obstacle": (139, 69, 19),          # Saddle Brown
            "text": (255, 255, 255),            # White
            "grid": (40, 40, 60),               # Dark Gray
            "performance_good": (0, 255, 0),    # Green
            "performance_medium": (255, 255, 0), # Yellow
            "performance_poor": (255, 0, 0)     # Red
        }
        
        # Fonts
        self.fonts = {
            "small": pygame.font.Font(None, 20),
            "medium": pygame.font.Font(None, 24),
            "large": pygame.font.Font(None, 32)
        }
        
        # View settings
        self.camera_offset = np.array([0.0, 0.0])
        self.zoom_level = 1.0
        self.show_grid = True
        self.show_metrics = True
        self.show_paths = False
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Initialize display only if not in headless mode
        if not self.headless:
            self._initialize_display()
    
    @property
    def is_headless(self) -> bool:
        """Check if the visualizer is running in headless mode."""
        return self.headless
    
    def _initialize_display(self):
        """Initialize the display and draw initial elements."""
        if not self.headless:
            self.screen.fill(self.colors["background"])
            pygame.display.flip()
    
    def update(self, step_result: Dict[str, Any]):
        """
        Update the visualization with new simulation data.
        
        Args:
            step_result: Result from the simulation step
        """
        if self.headless:
            return  # Skip rendering in headless mode
        
        # Handle events
        self._handle_events()
        
        # Clear screen
        self.screen.fill(self.colors["background"])
        
        # Calculate camera transform
        camera_transform = self._calculate_camera_transform()
        
        # Draw world elements
        self._draw_world(camera_transform)
        
        # Draw agents
        self._draw_agents(camera_transform)
        
        # Draw UI elements
        if self.show_metrics:
            self._draw_metrics(step_result)
        
        # Draw controls info
        self._draw_controls()
        
        # Update display
        pygame.display.flip()
        
        # Maintain FPS
        self._maintain_fps()
    
    def _handle_events(self):
        """Handle pygame events."""
        if self.headless:
            return  # Skip event handling in headless mode
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_m:
                    self.show_metrics = not self.show_metrics
                elif event.key == pygame.K_p:
                    self.show_paths = not self.show_paths
                elif event.key == pygame.K_r:
                    self._reset_camera()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_mouse_click(event.pos)
                elif event.button == 4:  # Mouse wheel up
                    self.zoom_level = min(3.0, self.zoom_level * 1.1)
                elif event.button == 5:  # Mouse wheel down
                    self.zoom_level = max(0.1, self.zoom_level / 1.1)
            
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button held
                    # Pan camera
                    self.camera_offset += np.array(event.rel) / self.zoom_level
    
    def _calculate_camera_transform(self) -> Dict[str, Any]:
        """Calculate camera transformation matrix."""
        # Calculate world center
        world_center = np.array([
            self.environment.world_size[0] / 2,
            self.environment.world_size[1] / 2
        ])
        
        # Calculate screen center
        screen_center = np.array([
            self.window_size[0] / 2,
            self.window_size[1] / 2
        ])
        
        # Calculate transform
        transform = {
            "offset": self.camera_offset,
            "zoom": self.zoom_level,
            "world_center": world_center,
            "screen_center": screen_center
        }
        
        return transform
    
    def _world_to_screen(self, world_pos: Tuple[float, float], 
                         transform: Dict[str, Any]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        world_pos = np.array(world_pos)
        
        # Apply zoom and offset
        screen_pos = (world_pos - transform["world_center"]) * transform["zoom"] + transform["screen_center"]
        screen_pos += transform["offset"]
        
        return (int(screen_pos[0]), int(screen_pos[1]))
    
    def _draw_world(self, transform: Dict[str, Any]):
        """Draw the world elements."""
        # Draw grid
        if self.show_grid:
            self._draw_grid(transform)
        
        # Draw obstacles
        for obstacle in self.environment.world.obstacles:
            screen_pos = self._world_to_screen((obstacle.x, obstacle.y), transform)
            radius = int(obstacle.radius * transform["zoom"])
            
            if 0 <= screen_pos[0] < self.window_size[0] and 0 <= screen_pos[1] < self.window_size[1]:
                pygame.draw.circle(self.screen, self.colors["obstacle"], screen_pos, radius)
        
        # Draw resources
        for resource in self.environment.world.resources:
            if not resource.collected:
                screen_pos = self._world_to_screen((resource.x, resource.y), transform)
                radius = int(8 * transform["zoom"])
                
                if 0 <= screen_pos[0] < self.window_size[0] and 0 <= screen_pos[1] < self.window_size[1]:
                    # Choose color based on resource type
                    color = self.colors.get(f"resource_{resource.resource_type}", self.colors["resource_food"])
                    pygame.draw.circle(self.screen, color, screen_pos, radius)
                    
                    # Draw resource value indicator
                    if transform["zoom"] > 0.5:
                        value_text = self.fonts["small"].render(str(int(resource.value)), True, self.colors["text"])
                        self.screen.blit(value_text, (screen_pos[0] + radius + 2, screen_pos[1] - 8))
        
        # Draw world border
        border_points = [
            (0, 0),
            (self.environment.world_size[0], 0),
            (self.environment.world_size[0], self.environment.world_size[1]),
            (0, self.environment.world_size[1])
        ]
        
        screen_border = [self._world_to_screen(point, transform) for point in border_points]
        pygame.draw.lines(self.screen, self.colors["world_border"], True, screen_border, 2)
    
    def _draw_grid(self, transform: Dict[str, Any]):
        """Draw the world grid."""
        grid_size = 100
        alpha = 50
        
        # Create a surface for the grid
        grid_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
        grid_surface.set_alpha(alpha)
        
        # Calculate grid lines
        start_x = int((transform["offset"][0] - transform["screen_center"][0]) / (grid_size * transform["zoom"]))
        end_x = int((transform["offset"][0] + self.window_size[0] - transform["screen_center"][0]) / (grid_size * transform["zoom"]))
        start_y = int((transform["offset"][1] - transform["screen_center"][1]) / (grid_size * transform["zoom"]))
        end_y = int((transform["offset"][1] + self.window_size[1] - transform["screen_center"][1]) / (grid_size * transform["zoom"]))
        
        # Draw vertical lines
        for x in range(start_x, end_x + 1):
            world_x = x * grid_size
            screen_x = self._world_to_screen((world_x, 0), transform)[0]
            if 0 <= screen_x < self.window_size[0]:
                pygame.draw.line(grid_surface, self.colors["grid"], 
                               (screen_x, 0), (screen_x, self.window_size[1]), 1)
        
        # Draw horizontal lines
        for y in range(start_y, end_y + 1):
            world_y = y * grid_size
            screen_y = self._world_to_screen((0, world_y), transform)[1]
            if 0 <= screen_y < self.window_size[1]:
                pygame.draw.line(grid_surface, self.colors["grid"], 
                               (0, screen_y), (self.window_size[0], screen_y), 1)
        
        self.screen.blit(grid_surface, (0, 0))
    
    def _draw_agents(self, transform: Dict[str, Any]):
        """Draw all agents in the swarm."""
        for agent in self.environment.swarm.agents:
            if agent.is_active():
                # Convert position to screen coordinates
                screen_pos = self._world_to_screen(agent.position, transform)
                radius = int(agent.radius * transform["zoom"])
                
                # Check if agent is visible on screen
                if (0 <= screen_pos[0] < self.window_size[0] and 
                    0 <= screen_pos[1] < self.window_size[1]):
                    
                    # Choose color based on agent type
                    color_key = f"agent_{agent.agent_type}"
                    color = self.colors.get(color_key, self.colors["agent_explorer"])
                    
                    # Draw agent body
                    pygame.draw.circle(self.screen, color, screen_pos, radius)
                    
                    # Draw agent border
                    pygame.draw.circle(self.screen, self.colors["text"], screen_pos, radius, 2)
                    
                    # Draw agent ID if zoomed in enough
                    if transform["zoom"] > 0.8:
                        id_text = self.fonts["small"].render(agent.agent_id.split("_")[1], True, self.colors["text"])
                        text_rect = id_text.get_rect(center=(screen_pos[0], screen_pos[1] - radius - 10))
                        self.screen.blit(id_text, text_rect)
                    
                    # Draw energy bar
                    if transform["zoom"] > 0.5:
                        self._draw_energy_bar(agent, screen_pos, radius)
                    
                    # Draw movement direction
                    if np.linalg.norm(agent.velocity) > 0.1:
                        direction_end = screen_pos + agent.velocity * transform["zoom"] * 2
                        pygame.draw.line(self.screen, self.colors["text"], screen_pos, 
                                       direction_end.astype(int), 2)
    
    def _draw_energy_bar(self, agent, screen_pos: Tuple[int, int], radius: int):
        """Draw energy bar above agent."""
        bar_width = radius * 2
        bar_height = 4
        bar_x = screen_pos[0] - bar_width // 2
        bar_y = screen_pos[1] - radius - 15
        
        # Background bar
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Energy bar
        energy_ratio = agent.energy / agent.max_energy
        energy_width = int(bar_width * energy_ratio)
        
        if energy_ratio > 0.6:
            energy_color = self.colors["performance_good"]
        elif energy_ratio > 0.3:
            energy_color = self.colors["performance_medium"]
        else:
            energy_color = self.colors["performance_poor"]
        
        pygame.draw.rect(self.screen, energy_color, 
                        (bar_x, bar_y, energy_width, bar_height))
    
    def _draw_metrics(self, step_result: Dict[str, Any]):
        """Draw performance metrics overlay."""
        # Get current metrics
        metrics = self.environment.get_environment_state()
        
        # Background panel
        panel_width = 300
        panel_height = 200
        panel_x = 10
        panel_y = 10
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw metrics
        y_offset = 20
        line_height = 25
        
        # Title
        title = self.fonts["medium"].render("Swarm Performance", True, self.colors["text"])
        self.screen.blit(title, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height + 5
        
        # Step information
        step_text = f"Step: {step_result.get('step', 0)}"
        step_surface = self.fonts["small"].render(step_text, True, self.colors["text"])
        self.screen.blit(step_surface, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height
        
        # Agent counts
        agent_text = f"Active Agents: {metrics['swarm_metrics']['active_agents']}"
        agent_surface = self.fonts["small"].render(agent_text, True, self.colors["text"])
        self.screen.blit(agent_surface, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height
        
        # Coordination score
        coord_score = metrics['swarm_metrics']['coordination_score']
        coord_text = f"Coordination: {coord_score:.3f}"
        coord_color = self._get_performance_color(coord_score)
        coord_surface = self.fonts["small"].render(coord_text, True, coord_color)
        self.screen.blit(coord_surface, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height
        
        # Task completion
        task_completion = metrics['task_statistics']['completion_rate']
        task_text = f"Task Completion: {task_completion:.3f}"
        task_color = self._get_performance_color(task_completion)
        task_surface = self.fonts["small"].render(task_text, True, task_color)
        self.screen.blit(task_surface, (panel_x + 10, panel_y + y_offset))
        y_offset += line_height
        
        # FPS
        if self.frame_times:
            fps = 1.0 / np.mean(self.frame_times)
            fps_text = f"FPS: {fps:.1f}"
            fps_surface = self.fonts["small"].render(fps_text, True, self.colors["text"])
            self.screen.blit(fps_surface, (panel_x + 10, panel_y + y_offset))
    
    def _get_performance_color(self, value: float) -> Tuple[int, int, int]:
        """Get color based on performance value."""
        if value > 0.7:
            return self.colors["performance_good"]
        elif value > 0.4:
            return self.colors["performance_medium"]
        else:
            return self.colors["performance_poor"]
    
    def _draw_controls(self):
        """Draw controls information."""
        controls = [
            "Controls:",
            "ESC - Quit",
            "G - Toggle Grid",
            "M - Toggle Metrics",
            "P - Toggle Paths",
            "R - Reset Camera",
            "Mouse - Pan & Zoom"
        ]
        
        y_offset = self.window_size[1] - 150
        for i, control in enumerate(controls):
            if i == 0:
                font = self.fonts["medium"]
                color = self.colors["text"]
            else:
                font = self.fonts["small"]
                color = (200, 200, 200)
            
            text_surface = font.render(control, True, color)
            self.screen.blit(text_surface, (self.window_size[0] - 200, y_offset + i * 20))
    
    def _handle_mouse_click(self, pos: Tuple[int, int]):
        """Handle mouse click events."""
        # Could implement agent selection, camera focus, etc.
        pass
    
    def _reset_camera(self):
        """Reset camera to default position."""
        self.camera_offset = np.array([0.0, 0.0])
        self.zoom_level = 1.0
    
    def _maintain_fps(self):
        """Maintain target FPS."""
        if self.headless:
            return  # Skip FPS maintenance in headless mode
            
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Calculate sleep time
        target_frame_time = 1.0 / self.fps
        if frame_time < target_frame_time:
            sleep_time = target_frame_time - frame_time
            time.sleep(sleep_time)
        
        self.last_frame_time = current_time
    
    def close(self):
        """Close the visualizer."""
        if not self.headless:
            pygame.quit()

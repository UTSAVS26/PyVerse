import random
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import time

def generate_maze(size):
    maze = [["#" for _ in range(size)] for _ in range(size)]
    
    def carve_passages(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx][ny] == "#":
                maze[x + dx // 2][y + dy // 2] = " "
                maze[nx][ny] = " "
                carve_passages(nx, ny)

    maze[1][1] = " "
    carve_passages(1, 1)
    maze[size - 2][size - 2] = " "
    
    for _ in range(size // 2):
        x, y = random.randint(1, size - 2), random.randint(1, size - 2)
        if maze[x][y] == " ":
            maze[x][y] = "#"

    return maze

def print_maze(maze, player_position):
    output = ""
    for i, row in enumerate(maze):
        line = ""
        for j, cell in enumerate(row):
            if (i, j) == player_position:
                line += "<span style='color: green;'>🧍</span>"
            else:
                line += "<span style='color: white;'>{}</span>".format(cell)
        output += line + "<br>"
    return output

class MazeGame:
    def __init__(self):
        self.size = 0
        self.maze = []
        self.position = (1, 1)
        self.exit_point = None
        self.move_limit = 0
        self.move_count = 0
        self.user_path = []
        self.desired_path = []
        self.output_widget = widgets.Output()
        self.start_game()
        
    def start_game(self):
        clear_output(wait=True)
        print("🎮 Select difficulty level: ")
        print("1. Intermediate (11x11)")
        print("2. Hard (15x15)")
        print("3. Very Hard (21x21)")
        
        self.difficulty_dropdown = widgets.Dropdown(
            options=['1', '2', '3'],
            description='Difficulty:',
        )
        self.start_button = widgets.Button(description="Start Game")
        self.start_button.on_click(self.set_difficulty)
        
        display(self.difficulty_dropdown, self.start_button, self.output_widget)

    def set_difficulty(self, b):
        if self.difficulty_dropdown.value == '1':
            self.size = 11
            self.move_limit = 20
        elif self.difficulty_dropdown.value == '2':
            self.size = 15
            self.move_limit = 30
        else:
            self.size = 21
            self.move_limit = 40
            
        self.maze = generate_maze(self.size)
        self.exit_point = (self.size - 2, self.size - 2)
        self.position = (1, 1)
        self.move_count = 0
        self.user_path = []
        self.desired_path = self.calculate_desired_path()
        
        self.play_game()

    def calculate_desired_path(self):
        from collections import deque
        
        queue = deque([(1, 1, [])])
        visited = set()
        visited.add((1, 1))

        while queue:
            x, y, path = queue.popleft()
            if (x, y) == self.exit_point:
                return path
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and
                    self.maze[nx][ny] == " " and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [(nx, ny)]))
        
        return []

    def play_game(self):
        clear_output(wait=True)
        self.start_time = time.time()
        with self.output_widget:
            display(HTML("<h2 style='color: blue;'>Welcome to the Maze Game!</h2>"))
            self.update_maze()
        
        button_box = widgets.HBox([
            widgets.Button(description="Up"),
            widgets.Button(description="Down"),
            widgets.Button(description="Left"),
            widgets.Button(description="Right"),
        ])
        
        for button in button_box.children:
            button.on_click(self.move_player)
        
        display(button_box)

    def move_player(self, button):
        direction = button.description.lower()
        new_position = self.position

        if direction == "up":
            new_position = (self.position[0] - 1, self.position[1])
        elif direction == "down":
            new_position = (self.position[0] + 1, self.position[1])
        elif direction == "left":
            new_position = (self.position[0], self.position[1] - 1)
        elif direction == "right":
            new_position = (self.position[0], self.position[1] + 1)

        if (0 <= new_position[0] < self.size and
            0 <= new_position[1] < self.size and
            self.maze[new_position[0]][new_position[1]] != "#"):
            self.position = new_position
            self.user_path.append(new_position)
            self.move_count += 1
            
            if self.position == self.exit_point:
                self.display_win_message()
            elif self.move_count >= self.move_limit:
                self.display_loss_message()
            else:
                self.update_maze()
        else:
            with self.output_widget:
                print("<span style='color: red;'>🚫 Invalid move! Can't go that way. Try again.</span>")

    def update_maze(self):
        with self.output_widget:
            clear_output(wait=True)
            print(HTML(print_maze(self.maze, self.position)))
            print(f"<span style='color: yellow;'>Moves left: {self.move_limit - self.move_count}</span>")
            print("<span style='color: cyan;'>Use the buttons to move.</span>")
    
    def display_win_message(self):
        clear_output(wait=True)
        score = self.calculate_score()
        with self.output_widget:
            print("<span style='color: green;'>🎉 Congratulations! You've reached the exit! 🎉</span>")
            print(f"<span style='color: white;'>Total moves: {self.move_count}</span>")
            print(f"<span style='color: gold;'>Your score: {score}</span>")
            self.start_game()

    def display_loss_message(self):
        clear_output(wait=True)
        with self.output_widget:
            print("<span style='color: red;'>🛑 Time's up or you've run out of moves! 🛑</span>")
            print(f"<span style='color: white;'>Total moves: {self.move_count}</span>")
            score = self.calculate_score()
            print(f"<span style='color: gold;'>Your score: {score}</span>")
            self.start_game()

    def calculate_score(self):
        correct_moves = len(set(self.user_path) & set(self.desired_path))
        total_moves = len(self.user_path)
        score = correct_moves * 10 - (total_moves - correct_moves) * 5
        return max(score, 0)

MazeGame()
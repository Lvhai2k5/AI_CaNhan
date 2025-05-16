import pygame
import sys
from pygame.locals import *
import copy
import time
import random
import tkinter as tk
from tkinter import ttk
import math
from collections import deque, defaultdict
import heapq

# Khởi tạo Pygame và kiểm tra
pygame.init()
if not pygame.get_init():
    print("Pygame initialization failed!")
    sys.exit(1)

# Cài đặt màn hình
WIDTH = 1080
HEIGHT = 600
CELL_SIZE = 120
GRID_SIZE = 3
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 50
CONTROL_PANEL_WIDTH = 600
CONTROL_PANEL_X = 470
screen = pygame.display.set_mode((WIDTH + 30, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver of Lê Vũ Hải")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (70, 130, 180)
LIGHT_BLUE = (173, 216, 230)
GREEN = (60, 179, 113)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
PURPLE = (147, 112, 219)
ORANGE = (255, 165, 0)
DARK_PURPLE = (106, 90, 205)
LIGHT_PURPLE = (186, 147, 255)
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
DARK_YELLOW = (218, 165, 32)
LIGHT_YELLOW = (255, 255, 102)
DARK_RED = (178, 34, 34)
LIGHT_RED = (255, 99, 71)
NEXT_MOVE_COLOR = (255, 182, 193)

# Font hỗ trợ tiếng Việt (Arial)
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 36)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 48)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 60)
    SMALL_FONT = pygame.font.Font(FONT_PATH, 24)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 36)
    TITLE_FONT = pygame.font.Font(None, 48)
    NUMBER_FONT = pygame.font.Font(None, 60)
    SMALL_FONT = pygame.font.Font(None, 24)

# Tải hình nền
picture_background = None
try:
    picture_background = pygame.image.load(r"download.png")
    picture_background = pygame.transform.scale(picture_background, (150, 150))
except:
    print("Không tìm thấy hình nền! Sử dụng màu trắng làm nền.")

# Trạng thái ban đầu và mục tiêu
INITIAL_STATE = [
    [1, 2, 3],
    [4, 0, 8],
    [5, 6, 7]
]
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Danh sách thuật toán
ALGORITHMS = ["BFS", "DFS", "IDs", "Simple Hill Climbing",
              "Stochastic Hill Climbing", "Steepest Hill Climbing",
              "Greedy", "UCS", "A*", "IDA*", "Simulated Annealing",
              "Genetic Algorithm", "Local Beam Search", "Q-Learning"]


def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_pos = [(x, y) for x in range(3) for y in range(3) if goal[x][y] == value][0]
                distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
    return distance


def get_neighbors(state):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        new_state = [row[:] for row in state]
                        new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                        neighbors.append(new_state)
                return neighbors
    return []


class DropdownMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chọn Thuật Toán")
        self.root.geometry("300x150")
        self.root.configure(bg="#f0f0f0")

        label = tk.Label(self.root, text="Chọn một thuật toán", font=("Arial", 12), bg="#f0f0f0")
        label.pack(pady=10)

        self.selected_algorithm = tk.StringVar()
        self.combobox = ttk.Combobox(
            self.root,
            textvariable=self.selected_algorithm,
            values=ALGORITHMS,
            state="readonly",
            font=("Arial", 10),
            width=30
        )
        self.combobox.set("BFS")
        self.combobox.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))
        self.button = ttk.Button(self.root, text="Xác nhận", command=self.root.quit)
        self.button.pack(pady=10)

    def get_selection(self):
        self.root.mainloop()
        return self.selected_algorithm.get()


class Puzzle:
    def __init__(self, initial_state):
        self.state = initial_state
        self.move_count = 0
        self.execution_time = 0

    def draw(self, screen, highlight_pos=None):
        for i in range(3):
            for j in range(3):
                x0 = j * CELL_SIZE + 50
                y0 = i * CELL_SIZE + 50
                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                if self.state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                elif highlight_pos and (i, j) == highlight_pos:
                    pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                if self.state[i][j] != 0:
                    text = NUMBER_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                    screen.blit(text, text_rect)

    def draw_best_individual(self, screen, state, x_offset, y_offset):
        small_cell_size = CELL_SIZE // 2
        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * small_cell_size
                y0 = y_offset + i * small_cell_size
                rect = pygame.Rect(x0, y0, small_cell_size, small_cell_size)
                if state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=5)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=5)
                pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=5)
                if state[i][j] != 0:
                    text = SMALL_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + small_cell_size // 2, y0 + small_cell_size // 2))
                    screen.blit(text, text_rect)

    def find_empty(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j
        return None

    def get_state_tuple(self):
        return tuple(tuple(row) for row in self.state)

    def is_solvable(self):
        """Check if the puzzle state is solvable by counting inversions"""
        flat_state = [self.state[i][j] for i in range(3) for j in range(3) if self.state[i][j] != 0]
        inversions = 0
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] > flat_state[j]:
                    inversions += 1
        return inversions % 2 == 0


class PuzzleSolver:
    def __init__(self, algorithm, puzzle):
        self.algorithm = algorithm
        self.puzzle = puzzle
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def q_learning(self, screen, font, small_font, selected_algorithm):
        """Q-learning algorithm to solve the 8-puzzle"""
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []

        # Initialize Q-table and parameters
        q_table = defaultdict(lambda: defaultdict(float))
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.3  # Exploration rate
        num_episodes = 1000
        max_steps = 100
        training_history = []

        def choose_action(state, valid_states):
            state_tuple = tuple(tuple(row) for row in state)
            valid_actions = []
            for i, next_state in enumerate(valid_states):
                if next_state != state:
                    valid_actions.append(i)
            if not valid_actions:
                return None, None
            if random.uniform(0, 1) < epsilon:
                return random.choice(valid_actions), valid_states[random.choice(valid_actions)]
            q_values = {i: q_table[state_tuple][i] for i in valid_actions}
            best_action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else random.choice(valid_actions)
            return best_action, valid_states[best_action]

        # Training phase
        for episode in range(num_episodes):
            state = [row[:] for row in self.puzzle.state]
            steps = 0
            total_reward = 0

            for _ in range(max_steps):
                neighbors = get_neighbors(state)
                if not neighbors:
                    break
                action, next_state = choose_action(state, neighbors)
                if action is None:
                    break
                next_state_tuple = tuple(tuple(row) for row in next_state)
                reward = 100 if next_state == GOAL_STATE else -manhattan_distance(next_state, GOAL_STATE)
                total_reward += reward

                # Update Q-value
                next_neighbors = get_neighbors(next_state)
                max_next_q = max([q_table[next_state_tuple][i] for i in range(len(next_neighbors))], default=0)
                state_tuple = tuple(tuple(row) for row in state)
                q_table[state_tuple][action] += alpha * (reward + gamma * max_next_q - q_table[state_tuple][action])

                state = next_state
                steps += 1
                if next_state == GOAL_STATE:
                    break

            epsilon = max(0.01, epsilon * 0.995)
            training_history.append({'episode': episode + 1, 'steps': steps, 'reward': total_reward})
            if (episode + 1) % 100 == 0:
                print(f"Q-Learning: Episode {episode + 1}/{num_episodes}, Steps: {steps}, Reward: {total_reward}")

        # Solving phase: Generate path using learned Q-table
        state = [row[:] for row in self.puzzle.state]
        path = [state]
        steps = 0
        visited = set()
        visited.add(tuple(tuple(row) for row in state))

        while steps < max_steps:
            neighbors = get_neighbors(state)
            if not neighbors:
                break
            state_tuple = tuple(tuple(row) for row in state)
            q_values = {i: q_table[state_tuple][i] for i in range(len(neighbors))}
            action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else random.choice(range(len(neighbors)))
            next_state = neighbors[action]
            next_state_tuple = tuple(tuple(row) for row in next_state)
            if next_state_tuple in visited:
                break
            visited.add(next_state_tuple)
            state = next_state
            path.append(state)
            self.puzzle.state = state
            if picture_background:
                screen.fill(WHITE)
                screen.blit(picture_background, (150, 450))
            else:
                screen.fill(WHITE)
            self.puzzle.draw(screen)
            draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
            pygame.display.flip()
            pygame.time.wait(1000)
            if state == GOAL_STATE:
                print("Q-Learning found solution!")
                return path, []
            steps += 1

        print("Q-Learning did not find exact solution, returning best path.")
        return path, []

    def bfs(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        queue = deque([(self.puzzle.state, [])])
        visited = {self.puzzle.get_state_tuple()}
        while queue:
            state, path = queue.popleft()
            if state == GOAL_STATE:
                return path + [state], []
            for neighbor in get_neighbors(state):
                neighbor_tuple = tuple(tuple(row) for row in neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor, path + [state]))
        return [], []

    def dfs(self, screen, font, small_font, selected_algorithm, depth_limit=30):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        start_state = self.puzzle.get_state_tuple()
        stack = [(self.puzzle.state, [self.puzzle.state])]
        visited = {start_state}
        while stack:
            current_state, path = stack.pop()
            if len(path) - 1 < depth_limit:
                for new_state in get_neighbors(current_state):
                    new_state_tuple = tuple(tuple(row) for row in new_state)
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        if new_state == GOAL_STATE:
                            return path + [new_state], []
                        stack.append((new_state, path + [new_state]))
        return [], []

    def iterative_deepening(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        depth = 0
        while True:
            path, _ = self.dfs(screen, font, small_font, selected_algorithm, depth)
            if path:
                return path, []
            depth += 1
            if depth > 30:
                return [], []
        return [], []

    def hill_climbing_simple(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            for neighbor in neighbors:
                neighbor_h = manhattan_distance(neighbor, GOAL_STATE)
                if neighbor_h < current_h:
                    current_state = neighbor
                    path.append(neighbor)
                    self.puzzle.state = current_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                    pygame.display.flip()
                    pygame.time.wait(1000)
                    break
            else:
                return path, []
        return path, []

    def hill_climbing_stochastic(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            better_neighbors = [(n, manhattan_distance(n, GOAL_STATE))
                                for n in neighbors if manhattan_distance(n, GOAL_STATE) < current_h]
            if better_neighbors:
                next_state = random.choice([n for n, _ in better_neighbors])
                current_state = next_state
                path.append(next_state)
                self.puzzle.state = current_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                pygame.display.flip()
                pygame.time.wait(1000)
            else:
                return path, []
        return path, []

    def hill_climbing_steepest(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        current_state = [row[:] for row in self.puzzle.state]
        path = [current_state]
        max_iterations = 100
        for _ in range(max_iterations):
            if current_state == GOAL_STATE:
                return path, []
            neighbors = get_neighbors(current_state)
            current_h = manhattan_distance(current_state, GOAL_STATE)
            best_neighbor = min(neighbors,
                                key=lambda x: manhattan_distance(x, GOAL_STATE),
                                default=None)
            best_h = manhattan_distance(best_neighbor, GOAL_STATE) if best_neighbor else float('inf')
            if best_h < current_h:
                current_state = best_neighbor
                path.append(best_neighbor)
                self.puzzle.state = current_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, [], 0)
                pygame.display.flip()
                pygame.time.wait(1000)
            else:
                return path, []
        return path, []

    def greedy(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        start_state = self.puzzle.get_state_tuple()
        heap = [(manhattan_distance(self.puzzle.state, GOAL_STATE), 0, self.puzzle.state, [self.puzzle.state])]
        visited = {start_state}
        while heap:
            _, _, current_state, path = heapq.heappop(heap)
            if current_state == GOAL_STATE:
                return path, []
            for new_state in get_neighbors(current_state):
                new_state_tuple = tuple(tuple(row) for row in new_state)
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    h = manhattan_distance(new_state, GOAL_STATE)
                    heapq.heappush(heap, (h, len(path), new_state, path + [new_state]))
        return [], []

    def uniform_cost(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        start_state = self.puzzle.get_state_tuple()
        heap = [(0, self.puzzle.state, [self.puzzle.state])]
        visited = {start_state}
        while heap:
            cost, current_state, path = heapq.heappop(heap)
            if current_state == GOAL_STATE:
                return path, []
            for new_state in get_neighbors(current_state):
                new_state_tuple = tuple(tuple(row) for row in new_state)
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    new_cost = len(path)
                    heapq.heappush(heap, (new_cost, new_state, path + [new_state]))
        return [], []

    def a_star(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []
        start_state = self.puzzle.get_state_tuple()
        heap = [(manhattan_distance(self.puzzle.state, GOAL_STATE), 0, self.puzzle.state, [self.puzzle.state])]
        visited = {start_state: 0}
        while heap:
            f, g, current_state, path = heapq.heappop(heap)
            if current_state == GOAL_STATE:
                return path, []
            for new_state in get_neighbors(current_state):
                new_state_tuple = tuple(tuple(row) for row in new_state)
                new_g = g + 1
                new_h = manhattan_distance(new_state, GOAL_STATE)
                new_f = new_g + new_h
                if new_state_tuple not in visited or new_g < visited[new_state_tuple]:
                    visited[new_state_tuple] = new_g
                    heapq.heappush(heap, (new_f, new_g, new_state, path + [new_state]))
        return [], []

    def ida_star(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []

        def search(path, g, threshold):
            current_state = path[-1]
            f = g + manhattan_distance(current_state, GOAL_STATE)
            if f > threshold:
                return f, None
            if current_state == GOAL_STATE:
                return f, path
            min_threshold = float('inf')
            for new_state in get_neighbors(current_state):
                new_state_tuple = tuple(tuple(row) for row in new_state)
                if new_state_tuple not in [tuple(tuple(row) for row in state) for state in path]:
                    path.append(new_state)
                    t, result = search(path, g + 1, threshold)
                    if result is not None:
                        return t, result
                    if t < min_threshold:
                        min_threshold = t
                    path.pop()
            return min_threshold, None

        threshold = manhattan_distance(self.puzzle.state, GOAL_STATE)
        path = [self.puzzle.state]
        while True:
            t, result = search(path, 0, threshold)
            if result is not None:
                return result, []
            if t == float('inf'):
                return [], []
            threshold = t
        return [], []

    def local_beam_search(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []

        def generate_individual():
            state = [row[:] for row in self.puzzle.state]
            for _ in range(random.randint(5, 20)):
                neighbors = get_neighbors(state)
                state = random.choice(neighbors)
            return state

        def fitness(state):
            return -manhattan_distance(state, GOAL_STATE)

        k = 10
        max_iterations = 1000
        stochastic_factor = 0.3
        states = [generate_individual() for _ in range(k)]
        path = [self.puzzle.state]
        best_individuals = []
        best_state = states[0]
        best_fitness = fitness(best_state)
        best_individuals.append((best_state, 0, best_fitness))
        iteration = 0

        while iteration < max_iterations:
            successors = []
            for state in states:
                neighbors = get_neighbors(state)
                successors.extend(neighbors)

            for successor in successors:
                if successor == GOAL_STATE:
                    path.append(successor)
                    best_individuals.append((successor, iteration, 0))
                    print(f"Local Beam Search found solution after {iteration} iterations")
                    self.puzzle.state = successor
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(5000)
                    return path, best_individuals

            successor_tuples = [tuple(tuple(row) for row in s) for s in successors]
            unique_successors = []
            seen = set()
            for s, t in zip(successors, successor_tuples):
                if t not in seen:
                    unique_successors.append(s)
                    seen.add(t)

            if not unique_successors:
                print("No more unique successors available.")
                break

            successor_fitness = [(s, fitness(s)) for s in unique_successors]
            fitness_scores = [f for _, f in successor_fitness]
            min_fitness = min(fitness_scores)
            normalized_fitness = [f - min_fitness + 1 for f in fitness_scores]
            total_fitness = sum(normalized_fitness)
            probabilities = [f / total_fitness for f in normalized_fitness] if total_fitness > 0 else [1 / len(
                normalized_fitness)] * len(normalized_fitness)

            new_states = []
            sorted_successors = sorted(successor_fitness, key=lambda x: x[1], reverse=True)
            top_k = min(k // 2, len(sorted_successors))
            for i in range(top_k):
                new_states.append(sorted_successors[i][0])

            remaining = k - len(new_states)
            if remaining > 0 and unique_successors:
                chosen = random.choices(
                    unique_successors,
                    weights=probabilities,
                    k=remaining
                )
                new_states.extend([s for s in chosen if s not in new_states])

            while len(new_states) < k and unique_successors:
                new_states.append(random.choice(unique_successors))

            states = new_states[:k]

            current_best = max(states, key=fitness, default=states[0])
            current_fitness = fitness(current_best)
            if current_fitness > best_fitness:
                best_state = [row[:] for row in current_best]
                best_fitness = current_fitness
                best_individuals.append((best_state, iteration, best_fitness))
                self.puzzle.state = best_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                        len(best_individuals) - 1)
                pygame.display.flip()
                pygame.time.wait(1000)

            path.append(states[0])
            iteration += 1

        print("Local Beam Search không tìm thấy giải pháp chính xác, trả về trạng thái tốt nhất.")
        path.append(best_state)
        best_individuals.append((best_state, iteration - 1, best_fitness))
        self.puzzle.state = best_state
        if picture_background:
            screen.fill(WHITE)
            screen.blit(picture_background, (150, 450))
        else:
            screen.fill(WHITE)
        self.puzzle.draw(screen)
        draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                len(best_individuals) - 1)
        pygame.display.flip()
        pygame.time.wait(5000)
        return path, best_individuals

    def simulated_annealing(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []

        initial_temp = 1000.0
        alpha = 0.995
        max_iterations = 10000
        current_state = [row[:] for row in self.puzzle.state]
        best_state = current_state
        best_cost = manhattan_distance(current_state, GOAL_STATE)
        path = [current_state]
        best_individuals = [(current_state, 0, -best_cost)]
        temp = initial_temp
        iteration = 0

        while iteration < max_iterations and temp > 0.1:
            neighbors = get_neighbors(current_state)
            next_state = random.choice(neighbors)
            current_cost = manhattan_distance(current_state, GOAL_STATE)
            next_cost = manhattan_distance(next_state, GOAL_STATE)
            delta_e = next_cost - current_cost

            if next_state == GOAL_STATE:
                path.append(next_state)
                best_individuals.append((next_state, iteration, 0))
                print(f"Simulated Annealing found solution after {iteration} iterations")
                self.puzzle.state = next_state
                if picture_background:
                    screen.fill(WHITE)
                    screen.blit(picture_background, (150, 450))
                else:
                    screen.fill(WHITE)
                self.puzzle.draw(screen)
                draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                        len(best_individuals) - 1)
                pygame.display.flip()
                pygame.time.wait(5000)
                return path, best_individuals

            if delta_e <= 0 or random.random() < math.exp(-delta_e / temp):
                current_state = [row[:] for row in next_state]
                path.append(current_state)
                if next_cost < best_cost:
                    best_state = [row[:] for row in next_state]
                    best_cost = next_cost
                    best_individuals.append((best_state, iteration, -best_cost))
                    self.puzzle.state = best_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(1000)

            temp *= alpha
            iteration += 1

        print("Simulated Annealing không tìm thấy giải pháp chính xác, trả về trạng thái tốt nhất.")
        path.append(best_state)
        best_individuals.append((best_state, iteration - 1, -best_cost))
        self.puzzle.state = best_state
        if picture_background:
            screen.fill(WHITE)
            screen.blit(picture_background, (150, 450))
        else:
            screen.fill(WHITE)
        self.puzzle.draw(screen)
        draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                len(best_individuals) - 1)
        pygame.display.flip()
        pygame.time.wait(5000)
        return path, best_individuals

    def genetic_algorithm(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            return [self.puzzle.state], []

        def generate_individual():
            state = [row[:] for row in self.puzzle.state]
            for _ in range(random.randint(5, 20)):
                neighbors = get_neighbors(state)
                state = random.choice(neighbors)
            return state

        def fitness(state):
            return -manhattan_distance(state, GOAL_STATE)

        def select_parents(population, fitness_scores):
            total_fitness = sum(fitness_scores)
            if total_fitness == 0:
                return random.choices(population, k=2)
            probabilities = [f / total_fitness for f in fitness_scores]
            return random.choices(population, weights=probabilities, k=2)

        def crossover(parent1, parent2):
            p1_flat = [parent1[i][j] for i in range(3) for j in range(3)]
            p2_flat = [parent2[i][j] for i in range(3) for j in range(3)]
            crossover_point = random.randint(1, 7)
            child1_flat = p1_flat[:crossover_point] + p2_flat[crossover_point:]
            child2_flat = p2_flat[:crossover_point] + p1_flat[crossover_point:]
            child1_flat = fix_duplicate(child1_flat)
            child2_flat = fix_duplicate(child2_flat)
            child1 = [[child1_flat[i * 3 + j] for j in range(3)] for i in range(3)]
            child2 = [[child2_flat[i * 3 + j] for j in range(3)] for i in range(3)]
            return child1, child2

        def fix_duplicate(flat_state):
            required = set(range(9))
            current = set(flat_state)
            missing = list(required - current)
            duplicates = []
            seen = set()
            for x in flat_state:
                if x in seen:
                    duplicates.append(x)
                else:
                    seen.add(x)
            random.shuffle(missing)
            for d in duplicates:
                idx = flat_state.index(d)
                flat_state[idx] = missing.pop()
            return flat_state

        def mutate(state, mutation_rate=0.1):
            if random.random() < mutation_rate:
                neighbors = get_neighbors(state)
                mutated_state = random.choice(neighbors)
                return mutated_state
            return state

        population_size = 50
        generations = 100
        mutation_rate = 0.1
        elite_size = max(1, population_size // 10)
        cull_threshold = -50

        population = [generate_individual() for _ in range(population_size)]
        path = [self.puzzle.state]
        best_individuals = []
        best_state = population[0]
        best_fitness = fitness(best_state)

        for gen in range(generations):
            fitness_scores = [fitness(state) for state in population]
            for state in population:
                if state == GOAL_STATE:
                    path.append(state)
                    best_individuals.append((state, gen, fitness(state)))
                    print(f"Genetic Algorithm found solution after {gen} generations")
                    self.puzzle.state = state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(5000)
                    return path, best_individuals
                if fitness(state) > best_fitness:
                    best_state = [row[:] for row in state]
                    best_fitness = fitness(state)
                    best_individuals.append((best_state, gen, best_fitness))
                    self.puzzle.state = best_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(1000)

            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                            :elite_size]
            new_population = [population[i] for i in elite_indices]

            population = [state for state, f in zip(population, fitness_scores) if f >= cull_threshold]
            if not population:
                population = [generate_individual() for _ in range(population_size - len(new_population))]
            else:
                fitness_scores = [fitness(state) for state in population]

            while len(new_population) < population_size:
                parent1, parent2 = select_parents(population, fitness_scores)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                child1_fitness = fitness(child1)
                child2_fitness = fitness(child2)
                if child1_fitness > best_fitness:
                    best_state = [row[:] for row in child1]
                    best_fitness = child1_fitness
                    best_individuals.append((best_state, gen, best_fitness))
                    self.puzzle.state = best_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(1000)
                if child2_fitness > best_fitness:
                    best_state = [row[:] for row in child2]
                    best_fitness = child2_fitness
                    best_individuals.append((best_state, gen, best_fitness))
                    self.puzzle.state = best_state
                    if picture_background:
                        screen.fill(WHITE)
                        screen.blit(picture_background, (150, 450))
                    else:
                        screen.fill(WHITE)
                    self.puzzle.draw(screen)
                    draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                            len(best_individuals) - 1)
                    pygame.display.flip()
                    pygame.time.wait(1000)
                new_population.extend([child1, child2])

            new_population = new_population[:population_size]
            population = new_population
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            best_individuals.append((population[best_idx], gen, fitness_scores[best_idx]))
            path.append(population[best_idx])
            print(f"Generation {gen}: Best fitness = {max(fitness_scores)}")

        print("Genetic Algorithm không tìm thấy giải pháp chính xác, trả về trạng thái tốt nhất.")
        path.append(best_state)
        best_individuals.append((best_state, generations - 1, best_fitness))
        self.puzzle.state = best_state
        if picture_background:
            screen.fill(WHITE)
            screen.blit(picture_background, (150, 450))
        else:
            screen.fill(WHITE)
        self.puzzle.draw(screen)
        draw_ui(screen, font, small_font, selected_algorithm, self.puzzle, True, False, best_individuals,
                len(best_individuals) - 1)
        pygame.display.flip()
        pygame.time.wait(5000)
        return path, best_individuals

    def solve(self, screen, font, small_font, selected_algorithm):
        start_time = time.time()
        print(f"Starting {self.algorithm}...")
        path = None
        best_individuals = []
        if self.algorithm == "BFS":
            path, best_individuals = self.bfs(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "DFS":
            path, best_individuals = self.dfs(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "IDs":
            path, best_individuals = self.iterative_deepening(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Simple Hill Climbing":
            path, best_individuals = self.hill_climbing_simple(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Stochastic Hill Climbing":
            path, best_individuals = self.hill_climbing_stochastic(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Steepest Hill Climbing":
            path, best_individuals = self.hill_climbing_steepest(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Greedy":
            path, best_individuals = self.greedy(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "UCS":
            path, best_individuals = self.uniform_cost(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "A*":
            path, best_individuals = self.a_star(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "IDA*":
            path, best_individuals = self.ida_star(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Local Beam Search":
            path, best_individuals = self.local_beam_search(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Simulated Annealing":
            path, best_individuals = self.simulated_annealing(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Genetic Algorithm":
            path, best_individuals = self.genetic_algorithm(screen, font, small_font, selected_algorithm)
        elif self.algorithm == "Q-Learning":
            path, best_individuals = self.q_learning(screen, font, small_font, selected_algorithm)
        else:
            print(f"Thuật toán {self.algorithm} chưa được triển khai.")
            path = []
            best_individuals = []
        end_time = time.time()
        self.puzzle.execution_time = end_time - start_time
        if path:
            print(f"{self.algorithm} completed in {self.puzzle.execution_time:.2f}s")
        return path, best_individuals


def draw_gradient_rect(screen, rect, color1, color2):
    x, y, w, h = rect
    gradient_surface = pygame.Surface((w, h))
    for i in range(h):
        ratio = i / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(gradient_surface, (r, g, b), (0, i), (w, i))
    screen.blit(gradient_surface, (x, y))


def draw_ui(screen, font, small_font, selected_algorithm, puzzle, is_running, is_paused, best_individuals,
            current_step):
    title_text = TITLE_FONT.render("Thông tin thuật toán", True, BLUE)
    screen.blit(title_text, (450, 20))
    shadow_rect = pygame.Rect(CONTROL_PANEL_X + 5, 85, CONTROL_PANEL_WIDTH, 480)
    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=15)
    pygame.draw.rect(screen, LIGHT_BLUE, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 480), border_radius=15)
    pygame.draw.rect(screen, DARK_GRAY, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 480), 3, border_radius=15)

    algo_text = font.render(f"Thuật toán: {selected_algorithm}", True, BLACK)
    algo_rect = algo_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 100))
    screen.blit(algo_text, algo_rect)

    steps_text = font.render(f"Số bước thực hiện: {puzzle.move_count}", True, BLACK)
    steps_rect = steps_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 150))
    screen.blit(steps_text, steps_rect)

    time_text = font.render(f"Thời gian: {puzzle.execution_time:.2f}s", True, BLACK)
    time_rect = time_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 200))
    screen.blit(time_text, time_rect)

    button_x = CONTROL_PANEL_X + (CONTROL_PANEL_WIDTH - BUTTON_WIDTH) // 2
    select_rect = pygame.Rect(button_x, 290, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, select_rect, DARK_PURPLE, LIGHT_PURPLE)
    pygame.draw.rect(screen, DARK_GRAY, select_rect, 3, border_radius=10)
    select_text = font.render("Thuật toán", True, WHITE)
    select_text_rect = select_text.get_rect(center=select_rect.center)
    screen.blit(select_text, select_text_rect)

    start_rect = pygame.Rect(button_x, 360, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 3, border_radius=10)
    start_text = font.render("Bắt đầu", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    pause_rect = pygame.Rect(button_x, 430, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_paused:
        draw_gradient_rect(screen, pause_rect, DARK_YELLOW, LIGHT_YELLOW)
    else:
        pygame.draw.rect(screen, ORANGE, pause_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, pause_rect, 3, border_radius=10)
    pause_text = font.render("Tạm Dừng" if not is_paused else "Tiếp Tục", True, WHITE)
    pause_text_rect = pause_text.get_rect(center=pause_rect.center)
    screen.blit(pause_text, pause_text_rect)

    reset_rect = pygame.Rect(button_x, 500, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 3, border_radius=10)
    reset_text = font.render("Đặt Lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    if (selected_algorithm in ["Genetic Algorithm", "Simulated Annealing", "Local Beam Search",
                               "Q-Learning"]) and best_individuals and current_step < len(best_individuals):
        best_state, step_or_gen, best_fitness = best_individuals[current_step]
        label = "Thế hệ" if selected_algorithm == "Genetic Algorithm" else "Lần lặp"
        best_text = small_font.render(f"Trạng thái tốt nhất ({label} {step_or_gen}, Fitness: {best_fitness})", True,
                                      BLACK)
        best_rect = best_text.get_rect(topleft=(CONTROL_PANEL_X + 20, 250))
        screen.blit(best_text, best_rect)
        puzzle.draw_best_individual(screen, best_state, CONTROL_PANEL_X + 20, 280)

    return select_rect, start_rect, pause_rect, reset_rect


def find_highlight_pos(puzzle, path, step):
    if step + 1 >= len(path):
        return None, None
    current = path[step]
    next_state = path[step + 1]
    empty_pos = None
    next_pos = None
    for i in range(3):
        for j in range(3):
            if current[i][j] == 0:
                empty_pos = (i, j)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3 and next_state[i][j] == current[ni][nj]:
                        next_pos = (ni, nj)
                        return empty_pos, next_pos
    return empty_pos, next_pos


def print_state(state, step):
    print(f"\nBước {step}:")
    for row in state:
        print(row)


def main():
    print("Starting main loop...")
    puzzle = Puzzle([row[:] for row in INITIAL_STATE])
    if not puzzle.is_solvable():
        print("Initial state is not solvable! Generating a solvable state...")
        state = [row[:] for row in GOAL_STATE]
        for _ in range(15):  # Medium difficulty, similar to previous code
            neighbors = get_neighbors(state)
            state = random.choice(neighbors)
        puzzle = Puzzle(state)
        print("New solvable initial state:")
        for row in state:
            print(row)
    selected_algorithm = "BFS"
    running = True
    solving = False
    paused = False
    path = []
    best_individuals = []
    step = 0
    clock = pygame.time.Clock()
    try:
        while running:
            if picture_background:
                screen.fill(WHITE)
                screen.blit(picture_background, (150, 450))
            else:
                screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    select_rect, start_rect, pause_rect, reset_rect = draw_ui(
                        screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, best_individuals, step
                    )
                    if select_rect.collidepoint(x, y) and not solving:
                        dropdown = DropdownMenu()
                        selected_algorithm = dropdown.get_selection()
                        dropdown.root.destroy()
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                        if not puzzle.is_solvable():
                            print("Initial state is not solvable! Generating a solvable state...")
                            state = [row[:] for row in GOAL_STATE]
                            for _ in range(15):
                                neighbors = get_neighbors(state)
                                state = random.choice(neighbors)
                            puzzle = Puzzle(state)
                            print("New solvable initial state:")
                            for row in state:
                                print(row)
                    if start_rect.collidepoint(x, y) and not solving:
                        solving = True
                        paused = False
                        solver = PuzzleSolver(selected_algorithm, puzzle)
                        path, best_individuals = solver.solve(screen, FONT, SMALL_FONT, selected_algorithm)
                        step = 0
                        puzzle.move_count = 0
                        if not path:
                            print("Không tìm thấy giải pháp!")
                            solving = False
                        else:
                            print("Đã tìm thấy giải pháp! Bắt đầu in các trạng thái:")
                    if pause_rect.collidepoint(x, y) and solving:
                        paused = not paused
                    if reset_rect.collidepoint(x, y):
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                        if not puzzle.is_solvable():
                            print("Initial state is not solvable! Generating a solvable state...")
                            state = [row[:] for row in GOAL_STATE]
                            for _ in range(15):
                                neighbors = get_neighbors(state)
                                state = random.choice(neighbors)
                            puzzle = Puzzle(state)
                            print("New solvable initial state:")
                            for row in state:
                                print(row)
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        solving = False
                        paused = False
                        path = []
                        best_individuals = []
                        step = 0
                        print("Đã đặt lại trạng thái ban đầu:")
            if solving and not paused and path and step < len(path):
                empty_pos, next_pos = find_highlight_pos(puzzle, path, step)
                puzzle.state = [row[:] for row in path[step]]
                puzzle.move_count = step
                print_state(puzzle.state, step)
                step += 1
                pygame.time.wait(1000)
                if step >= len(path):
                    solving = False
                    print("Đã hoàn thành giải pháp!")
            empty_pos, next_pos = find_highlight_pos(puzzle, path, step - 1) if path and step > 0 else (None, None)
            puzzle.draw(screen, next_pos if next_pos and solving else None)
            draw_ui(screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, best_individuals,
                    step - 1 if step > 0 else 0)
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
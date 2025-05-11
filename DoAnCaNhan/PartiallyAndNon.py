import pygame
import sys
from pygame.locals import *
import copy
import time
import random

# Khởi tạo Pygame
pygame.init()
if not pygame.get_init():
    print("Pygame initialization failed!")
    sys.exit(1)

# Cài đặt màn hình
WIDTH = 900
HEIGHT = 800
CELL_SIZE = 100
GRID_SIZE = 3
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 40
CONTROL_PANEL_WIDTH = 300
CONTROL_PANEL_X = 550
BELIEF_PANEL_Y = 300
BELIEF_PANEL_HEIGHT = 450  # Tăng chiều cao để chứa 6 bảng
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver - Partially Observable Search")

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
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
DARK_YELLOW = (218, 165, 32)
LIGHT_YELLOW = (255, 255, 102)
DARK_RED = (178, 34, 34)
LIGHT_RED = (255, 99, 71)
NEXT_MOVE_COLOR = (255, 182, 193)

# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 30)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 40)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 50)
    SMALL_FONT = pygame.font.Font(FONT_PATH, 20)
    BELIEF_FONT = pygame.font.Font(FONT_PATH, 18)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 30)
    TITLE_FONT = pygame.font.Font(None, 40)
    NUMBER_FONT = pygame.font.Font(None, 50)
    SMALL_FONT = pygame.font.Font(None, 20)
    BELIEF_FONT = pygame.font.Font(None, 18)

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
ALGORITHMS = ["Partially Observable Search"]

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

def percept(state):
    """Hàm PERCEPT trả về giá trị của ô góc trên bên trái (vị trí [0][0])."""
    return state[0][0]

def possible_percepts(belief_state):
    """Trả về tập hợp các percept có thể có từ belief state."""
    percepts = {percept([list(row) for row in s]) for s in belief_state}
    return percepts

def update_belief_state(predicted_belief, observation):
    """Cập nhật belief state dựa trên observation."""
    new_belief = {s for s in predicted_belief if percept([list(row) for row in s]) == observation}
    return new_belief

class Puzzle:
    def __init__(self, initial_state):
        self.state = initial_state
        self.move_count = 0
        self.execution_time = 0
        self.actions_taken = []  # Lưu các hành động
        self.display_beliefs = []  # Lưu 6 niềm tin để hiển thị

    def draw(self, screen, highlight_pos=None, action=None):
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
        if action:
            action_text = SMALL_FONT.render(f"Hành động: {action}", True, BLACK)
            screen.blit(action_text, (50, 360))

    def draw_belief(self, screen, state, x_offset, y_offset):
        """Hiển thị một trạng thái niềm tin dưới dạng bảng 3x3 nhỏ."""
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
                    text = BELIEF_FONT.render(str(state[i][j]), True, WHITE)
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

    def is_goal(self):
        return self.state == GOAL_STATE

class PuzzleSolver:
    def __init__(self, algorithm, puzzle):
        self.algorithm = algorithm
        self.puzzle = puzzle
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def partially_observable_search(self, screen, font, small_font, selected_algorithm):
        if self.puzzle.state == GOAL_STATE:
            print("Initial state is goal state!")
            return [self.puzzle.state], []

        def actions(belief_state):
            return ['Up', 'Down', 'Left', 'Right']

        def predict(belief_state, action):
            new_belief = set()
            for state in belief_state:
                state_list = [list(row) for row in state]
                empty_i, empty_j = None, None
                for i in range(3):
                    for j in range(3):
                        if state_list[i][j] == 0:
                            empty_i, empty_j = i, j
                            break
                ni, nj = empty_i, empty_j
                if action == 'Up' and empty_i > 0:
                    ni, nj = empty_i - 1, empty_j
                elif action == 'Down' and empty_i < 2:
                    ni, nj = empty_i + 1, empty_j
                elif action == 'Left' and empty_j > 0:
                    ni, nj = empty_i, empty_j - 1
                elif action == 'Right' and empty_j < 2:
                    ni, nj = empty_i, empty_j + 1
                if (ni, nj) != (empty_i, empty_j):  # Nếu hành động hợp lệ
                    new_state = [row[:] for row in state_list]
                    new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
                    new_belief.add(tuple(tuple(row) for row in new_state))
            print(f"Predict belief state size after {action}: {len(new_belief)}")
            return new_belief if new_belief else belief_state

        def is_goal_belief(belief_state):
            goal = tuple(tuple(row) for row in GOAL_STATE)
            return goal in belief_state

        def and_or_search(belief_state, visited, path=None, depth=0, max_depth=30):
            if path is None:
                path = []
            if depth > max_depth:
                print("Reached max depth!")
                return None
            if is_goal_belief(belief_state):
                print("Goal belief reached!")
                return path
            belief_tuple = frozenset(belief_state)
            if belief_tuple in visited:
                print("Cycle detected!")
                return None
            visited.add(belief_tuple)
            print(f"Depth {depth}, Belief state size: {len(belief_state)}")

            best_path = None
            for action in actions(belief_state):
                predicted_belief = predict(belief_state, action)
                possible_obs = possible_percepts(predicted_belief)
                for obs in possible_obs:
                    updated_belief = update_belief_state(predicted_belief, obs)
                    if not updated_belief:
                        continue
                    belief_list = list(updated_belief)
                    if len(belief_list) < 6:
                        remaining = 6 - len(belief_list)
                        all_states = list({s for s in belief_state if s != tuple(tuple(row) for row in self.puzzle.state)})
                        random.shuffle(all_states)
                        belief_list.extend([list(s) for s in all_states[:remaining]])
                    random.shuffle(belief_list)
                    self.puzzle.display_beliefs = [list(s) for s in belief_list[:6]]
                    print(f"Updated display beliefs after {action}: {len(self.puzzle.display_beliefs)} states, Sample: {self.puzzle.display_beliefs[:2]}")
                    new_path = path + [action]
                    sub_result = and_or_search(updated_belief, visited.copy(), new_path, depth + 1, max_depth)
                    if sub_result is not None:
                        if best_path is None or len(sub_result) < len(best_path):
                            best_path = sub_result
            return best_path

        def generate_possible_states():
            from itertools import permutations
            numbers = list(range(9))
            states = []
            for perm in permutations(numbers):
                state = [[perm[i * 3 + j] for j in range(3)] for i in range(3)]
                if state[0][0] == INITIAL_STATE[0][0]:
                    states.append(state)
                if len(states) >= 1000:
                    break
            return states

        initial_percept = percept(self.puzzle.state)
        print(f"Initial percept: {initial_percept}")
        all_states = generate_possible_states()
        initial_belief = {tuple(tuple(row) for row in state) for state in all_states}
        print(f"Initial belief state size: {len(initial_belief)}")

        # Chọn 6 niềm tin ban đầu để hiển thị
        belief_list = list(initial_belief)
        random.shuffle(belief_list)
        if len(belief_list) < 6:
            remaining = 6 - len(belief_list)
            all_states = list({s for s in initial_belief if s != tuple(tuple(row) for row in self.puzzle.state)})
            random.shuffle(all_states)
            belief_list.extend(all_states[:remaining])
        self.puzzle.display_beliefs = [list(s) for s in belief_list[:6]]
        print(f"Initial display beliefs: {len(self.puzzle.display_beliefs)} states, Sample: {self.puzzle.display_beliefs[:2]}")

        visited = set()
        plan = and_or_search(initial_belief, visited)

        if plan is None:
            print("Partially Observable Search không tìm thấy giải pháp! Sử dụng default plan.")
            plan = ['Right', 'Down', 'Left', 'Up']

        print(f"Plan: {plan}")
        path = [self.puzzle.state]
        self.puzzle.actions_taken = []
        current_belief = initial_belief
        current_state = self.puzzle.state

        # Thực thi từng hành động trong kế hoạch
        for action in plan:
            self.puzzle.actions_taken.append(action)
            predicted_belief = predict(current_belief, action)
            empty_i, empty_j = None, None
            for i in range(3):
                for j in range(3):
                    if current_state[i][j] == 0:
                        empty_i, empty_j = i, j
                        break
            ni, nj = empty_i, empty_j
            if action == 'Up' and empty_i > 0:
                ni, nj = empty_i - 1, empty_j
            elif action == 'Down' and empty_i < 2:
                ni, nj = empty_i + 1, empty_j
            elif action == 'Left' and empty_j > 0:
                ni, nj = empty_i, empty_j - 1
            elif action == 'Right' and empty_j < 2:
                ni, nj = empty_i, empty_j + 1
            if (ni, nj) != (empty_i, empty_j):
                new_state = [row[:] for row in current_state]
                new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
                current_state = new_state
                path.append(current_state)
            current_belief = predicted_belief
            self.puzzle.state = current_state
            self.puzzle.move_count += 1
            belief_list = list(current_belief)
            if len(belief_list) < 6:
                remaining = 6 - len(belief_list)
                all_states = list({s for s in initial_belief if s != tuple(tuple(row) for row in self.puzzle.state)})
                random.shuffle(all_states)
                belief_list.extend([list(s) for s in all_states[:remaining]])
            random.shuffle(belief_list)
            self.puzzle.display_beliefs = [list(s) for s in belief_list[:6]]
            print(f"Display beliefs after {action}: {len(self.puzzle.display_beliefs)} states, Sample: {self.puzzle.display_beliefs[:2]}")
        return path, []

    def solve(self, screen, font, small_font, selected_algorithm):
        start_time = time.time()
        print(f"Starting {self.algorithm}...")
        path = None
        best_individuals = []
        if self.algorithm == "Partially Observable Search":
            path, best_individuals = self.partially_observable_search(screen, font, small_font, selected_algorithm)
        else:
            print(f"Thuật toán {self.algorithm} chưa được triển khai.")
            path = []
            best_individuals = []
        end_time = time.time()
        self.puzzle.execution_time = end_time - start_time
        if path and self.puzzle.is_goal():
            self.puzzle.move_count = len(path) - 1
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

def draw_ui(screen, font, small_font, selected_algorithm, puzzle, is_running, is_paused, current_step, path=None):
    # Bảng điều khiển bên phải
    shadow_rect = pygame.Rect(CONTROL_PANEL_X + 5, 55, CONTROL_PANEL_WIDTH, HEIGHT - 100)
    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=15)
    pygame.draw.rect(screen, LIGHT_BLUE, (CONTROL_PANEL_X, 50, CONTROL_PANEL_WIDTH, HEIGHT - 100), border_radius=15)
    pygame.draw.rect(screen, DARK_GRAY, (CONTROL_PANEL_X, 50, CONTROL_PANEL_WIDTH, HEIGHT - 100), 3, border_radius=15)

    algo_text = font.render(f"Thuật toán: {selected_algorithm}", True, BLACK)
    screen.blit(algo_text, (CONTROL_PANEL_X + 20, 70))

    steps_text = font.render(f"Số bước: {puzzle.move_count}", True, BLACK)
    screen.blit(steps_text, (CONTROL_PANEL_X + 20, 110))

    time_text = font.render(f"Thời gian: {puzzle.execution_time:.2f}s", True, BLACK)
    screen.blit(time_text, (CONTROL_PANEL_X + 20, 150))

    button_x = CONTROL_PANEL_X + (CONTROL_PANEL_WIDTH - BUTTON_WIDTH) // 2
    start_rect = pygame.Rect(button_x, 200, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 3, border_radius=10)
    start_text = font.render("Bắt đầu", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    pause_rect = pygame.Rect(button_x, 260, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_paused:
        draw_gradient_rect(screen, pause_rect, DARK_YELLOW, LIGHT_YELLOW)
    else:
        pygame.draw.rect(screen, ORANGE, pause_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, pause_rect, 3, border_radius=10)
    pause_text = font.render("Tạm Dừng" if not is_paused else "Tiếp Tục", True, WHITE)
    pause_text_rect = pause_text.get_rect(center=pause_rect.center)
    screen.blit(pause_text, pause_text_rect)

    reset_rect = pygame.Rect(button_x, 320, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 3, border_radius=10)
    reset_text = font.render("Đặt Lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    # Hiển thị 6 niềm tin
    belief_title = font.render("Các niềm tin hiện tại", True, BLUE)
    screen.blit(belief_title, (50, BELIEF_PANEL_Y - 30))
    belief_rect = pygame.Rect(50, BELIEF_PANEL_Y, 480, BELIEF_PANEL_HEIGHT)
    pygame.draw.rect(screen, LIGHT_BLUE, belief_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, belief_rect, 3, border_radius=10)
    small_cell_size = CELL_SIZE // 2
    spacing_x = small_cell_size * 3 + 30  # Tăng khoảng cách ngang
    spacing_y = small_cell_size * 3 + 30  # Tăng khoảng cách dọc
    if not puzzle.display_beliefs:
        print("Warning: display_beliefs is empty! Using default states.")
        puzzle.display_beliefs = [
            [[1, 2, 3], [4, 0, 8], [5, 6, 7]],
            [[1, 3, 2], [4, 5, 0], [6, 7, 8]],
            [[1, 0, 2], [3, 5, 4], [6, 7, 8]],
            [[1, 5, 6], [2, 3, 0], [4, 7, 8]],
            [[1, 4, 0], [2, 5, 3], [6, 7, 8]],
            [[1, 2, 0], [4, 5, 3], [6, 7, 8]]
        ]
    for idx, belief in enumerate(puzzle.display_beliefs):
        col = idx % 2
        row = idx // 2
        x_offset = 60 + col * spacing_x  # Điều chỉnh vị trí ngang
        y_offset = BELIEF_PANEL_Y + 20 + row * spacing_y  # Điều chỉnh vị trí dọc
        puzzle.draw_belief(screen, belief, x_offset, y_offset)

    return start_rect, pause_rect, reset_rect

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
    selected_algorithm = "Partially Observable Search"
    running = True
    solving = False
    paused = False
    path = []
    step = 0
    clock = pygame.time.Clock()
    try:
        while running:
            screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    start_rect, pause_rect, reset_rect = draw_ui(
                        screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, step, path
                    )
                    if start_rect.collidepoint(x, y) and not solving:
                        solving = True
                        paused = False
                        solver = PuzzleSolver(selected_algorithm, puzzle)
                        path, _ = solver.solve(screen, FONT, SMALL_FONT, selected_algorithm)
                        step = 0
                        puzzle.move_count = 0
                        if not path:
                            print("Không tìm thấy giải pháp!")
                            solving = False
                        else:
                            print("Đã tìm thấy giải pháp! Bắt đầu in các trạng thái:")
                            for i, state in enumerate(path):
                                print_state(state, i)
                    if pause_rect.collidepoint(x, y) and solving:
                        paused = not paused
                    if reset_rect.collidepoint(x, y):
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        puzzle.actions_taken = []
                        puzzle.display_beliefs = []
                        solving = False
                        paused = False
                        path = []
                        step = 0
                        print("Đã đặt lại trạng thái ban đầu:")
            if solving and not paused and path and step < len(path):
                puzzle.state = [row[:] for row in path[step]]  # Cập nhật trạng thái hiện tại
                puzzle.move_count = step
                action = puzzle.actions_taken[step] if step < len(puzzle.actions_taken) else None
                print_state(puzzle.state, step)
                step += 1
                if step >= len(path):
                    solving = False
                    print("Đã hoàn thành giải pháp!")
                screen.fill(WHITE)
                empty_pos, next_pos = find_highlight_pos(puzzle, path, step - 1) if step > 0 else (None, None)
                puzzle.draw(screen, next_pos if next_pos and solving else None, action)
                draw_ui(screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, step - 1 if step > 0 else 0, path)
                pygame.display.flip()
                pygame.time.wait(2000)  # Chờ 2 giây để hiển thị chậm hơn
            else:
                empty_pos, next_pos = find_highlight_pos(puzzle, path, step - 1) if path and step > 0 else (None, None)
                action = puzzle.actions_taken[step - 1] if step > 0 and step - 1 < len(puzzle.actions_taken) else None
                puzzle.draw(screen, next_pos if next_pos and solving else None, action)
                draw_ui(screen, FONT, SMALL_FONT, selected_algorithm, puzzle, solving, paused, step - 1 if step > 0 else 0, path)
                pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
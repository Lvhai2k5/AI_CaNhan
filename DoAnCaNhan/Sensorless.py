import pygame
import sys
from pygame.locals import *
import copy
from collections import deque
import time
import math

# Khởi tạo Pygame
pygame.init()

# Cài đặt màn hình
WIDTH = 1280
HEIGHT = 720
CELL_SIZE = 80
GRID_SIZE = 3
BUTTON_WIDTH = 160
BUTTON_HEIGHT = 45
CONTROL_PANEL_WIDTH = 350
CONTROL_PANEL_HEIGHT = 250
CONTROL_PANEL_X = (WIDTH - CONTROL_PANEL_WIDTH) // 2
CONTROL_PANEL_Y = 20  # Bảng điều khiển ở trên cùng
GRID_OFFSET_X = 30
GRID_OFFSET_Y = CONTROL_PANEL_Y + CONTROL_PANEL_HEIGHT + 100  # Tăng khoảng cách lên 100 để dịch xuống dưới
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver - Sensorless")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
DARK_GRAY = (40, 40, 40)
BLUE = (50, 100, 150)
LIGHT_BLUE = (135, 206, 235)
GREEN = (50, 150, 50)
RED = (200, 50, 50)
YELLOW = (255, 200, 0)
DARK_GREEN = (20, 120, 20)
LIGHT_GREEN = (100, 255, 100)
DARK_YELLOW = (200, 150, 0)
LIGHT_YELLOW = (255, 255, 150)
DARK_RED = (150, 30, 30)
LIGHT_RED = (255, 80, 80)
NEXT_MOVE_COLOR = (255, 150, 150)
GOAL_COLOR = (144, 238, 144)
BG_GRADIENT_TOP = (200, 220, 255)
BG_GRADIENT_BOTTOM = (255, 255, 255)
PANEL_BG = (240, 248, 255)

# Font hỗ trợ tiếng Việt
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 24)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 32)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 36)
    BOLD_FONT = pygame.font.Font(FONT_PATH, 48)
    SMALL_FONT = pygame.font.Font(FONT_PATH, 20)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 24)
    TITLE_FONT = pygame.font.Font(None, 32)
    NUMBER_FONT = pygame.font.Font(None, 36)
    BOLD_FONT = pygame.font.Font(None, 48)
    SMALL_FONT = pygame.font.Font(None, 20)

# Trạng thái mục tiêu
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Ba trạng thái niềm tin ban đầu
BELIEF_STATE_1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 0, 8]
]
BELIEF_STATE_2 = [
    [1, 2, 3],
    [4, 5, 6],
    [0, 7, 8]
]
BELIEF_STATE_3 = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]


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


class Puzzle:
    def __init__(self, belief_states):
        self.belief_states = belief_states
        self.move_count = 0
        self.execution_time = 0
        self.error_message = ""
        self.animations = [{} for _ in belief_states]

    def start_animation(self, belief_idx, current_state, next_state):
        if current_state == GOAL_STATE:
            print(f"Belief {belief_idx + 1} already at goal, skipping animation")
            return
        empty_i, empty_j = self.find_empty_in_state(current_state)
        if empty_i is None or empty_j is None:
            print(f"Error: No empty tile in Belief {belief_idx + 1}")
            return
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = empty_i + di, empty_j + dj
            if 0 <= ni < 3 and 0 <= nj < 3 and next_state[empty_i][empty_j] == current_state[ni][nj]:
                tile_value = current_state[ni][nj]
                self.animations[belief_idx][tile_value] = {
                    'start': (ni, nj),
                    'end': (empty_i, empty_j),
                    'progress': 0.0,
                    'duration': 0.5
                }
                print(
                    f"Started animation for Belief {belief_idx + 1}, tile {tile_value}: {ni},{nj} -> {empty_i},{empty_j}")
                break

    def update_animations(self, dt):
        for belief_idx, animation in enumerate(self.animations):
            for tile_value in list(animation.keys()):
                anim = animation[tile_value]
                anim['progress'] += dt / anim['duration']
                if anim['progress'] >= 1.0:
                    del animation[tile_value]
                    print(f"Animation completed for Belief {belief_idx + 1}, tile {tile_value}")
                else:
                    animation[tile_value] = anim

    def draw(self, screen, highlight_positions=None, show_goal_only=False, animation_time=0):
        active_states = sum(1 for state in self.belief_states if state != GOAL_STATE)
        all_at_goal = active_states == 0

        if show_goal_only or all_at_goal:
            pulse_scale = 1.0 + 0.1 * math.sin(animation_time * 2)
            scaled_cell_size = int(CELL_SIZE * pulse_scale)
            x_offset = WIDTH // 2 - scaled_cell_size * 1.5
            y_offset = GRID_OFFSET_Y + 50  # Dịch xuống thêm để không che bảng điều khiển
            label_text = BOLD_FONT.render("Hoàn Thành Tất Cả!", True, RED)
            label_rect = label_text.get_rect(center=(WIDTH // 2, y_offset - 60))
            screen.blit(label_text, label_rect)
            for i in range(3):
                for j in range(3):
                    x0 = j * scaled_cell_size + x_offset
                    y0 = i * scaled_cell_size + y_offset
                    rect = pygame.Rect(x0, y0, scaled_cell_size, scaled_cell_size)
                    pygame.draw.rect(screen, GOAL_COLOR, rect, border_radius=8)
                    pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=8)
                    if GOAL_STATE[i][j] != 0:
                        text = NUMBER_FONT.render(str(GOAL_STATE[i][j]), True, WHITE)
                        text_rect = text.get_rect(center=(x0 + scaled_cell_size // 2, y0 + scaled_cell_size // 2))
                        screen.blit(text, text_rect)
        else:
            for idx, state in enumerate(self.belief_states):
                # Sắp xếp lưới theo hàng ngang, 3 lưới trên 1 hàng
                x_offset = GRID_OFFSET_X + (idx % 3) * (CELL_SIZE * 3 + 20)
                y_offset = GRID_OFFSET_Y
                if state == GOAL_STATE:
                    label_text = TITLE_FONT.render(f"Niềm tin {idx + 1} - Hoàn thành", True, GREEN)
                    label_rect = label_text.get_rect(center=(x_offset + CELL_SIZE * 1.5, y_offset - 30))
                    screen.blit(label_text, label_rect)
                    for i in range(3):
                        for j in range(3):
                            x0 = j * CELL_SIZE + x_offset
                            y0 = i * CELL_SIZE + y_offset
                            rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                            pygame.draw.rect(screen, GOAL_COLOR, rect, border_radius=8)
                            pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=8)
                            if GOAL_STATE[i][j] != 0:
                                text = NUMBER_FONT.render(str(GOAL_STATE[i][j]), True, WHITE)
                                text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                                screen.blit(text, text_rect)
                else:
                    label_text = TITLE_FONT.render(f"Niềm tin {idx + 1}", True, BLUE)
                    label_rect = label_text.get_rect(center=(x_offset + CELL_SIZE * 1.5, y_offset - 30))
                    screen.blit(label_text, label_rect)
                    for i in range(3):
                        for j in range(3):
                            tile_value = state[i][j]
                            if tile_value == 0:
                                x0 = j * CELL_SIZE + x_offset
                                y0 = i * CELL_SIZE + y_offset
                                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                                pygame.draw.rect(screen, GRAY, rect, border_radius=8)
                                pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=8)
                                continue
                            if tile_value in self.animations[idx]:
                                continue
                            x0 = j * CELL_SIZE + x_offset
                            y0 = i * CELL_SIZE + y_offset
                            rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                            if highlight_positions and idx in highlight_positions and (i, j) in highlight_positions[
                                idx]:
                                pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=8)
                            else:
                                pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=8)
                            pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=8)
                            text = NUMBER_FONT.render(str(tile_value), True, WHITE)
                            text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                            screen.blit(text, text_rect)
                    for tile_value, anim in self.animations[idx].items():
                        progress = min(anim['progress'], 1.0)
                        si, sj = anim['start']
                        ei, ej = anim['end']
                        x = sj + (ej - sj) * progress
                        y = si + (ei - si) * progress
                        x0 = x * CELL_SIZE + x_offset
                        y0 = y * CELL_SIZE + y_offset
                        rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=8)
                        pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=8)
                        text = NUMBER_FONT.render(str(tile_value), True, WHITE)
                        text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                        screen.blit(text, text_rect)
        if self.error_message:
            error_text = FONT.render(self.error_message, True, RED)
            error_rect = error_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))
            screen.blit(error_text, error_rect)

    def find_empty_in_state(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None, None


class PuzzleSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle

    def sensorless_bfs(self):
        print("Starting sensorless BFS...")
        if all(state == GOAL_STATE for state in self.puzzle.belief_states):
            print("All belief states are already at goal!")
            return [self.puzzle.belief_states]

        initial_belief = tuple(tuple(tuple(row) for row in state) for state in self.puzzle.belief_states)
        queue = deque([(self.puzzle.belief_states, [self.puzzle.belief_states])])
        visited = {initial_belief}
        max_iterations = 100
        iteration = 0

        while queue:
            if iteration >= max_iterations:
                print("Exceeded max iterations!")
                self.puzzle.error_message = "Đã vượt quá số lần lặp tối đa."
                return None
            iteration += 1

            current_belief, path = queue.popleft()
            print(f"Iteration {iteration}, Belief states: {len(current_belief)}")
            actions = set()
            for state in current_belief:
                empty_i, empty_j = self.puzzle.find_empty_in_state(state)
                if empty_i is None or empty_j is None:
                    print("Error: No empty tile found in state!")
                    self.puzzle.error_message = "Lỗi: Không tìm thấy ô trống!"
                    return None
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = empty_i + di, empty_j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        actions.add((di, dj))

            if not actions:
                print("No valid actions available!")
                continue

            for action in actions:
                new_belief = []
                for idx, state in enumerate(current_belief):
                    empty_i, empty_j = self.puzzle.find_empty_in_state(state)
                    ni, nj = empty_i + action[0], empty_j + action[1]
                    if not (0 <= ni < 3 and 0 <= nj < 3):
                        new_belief.append(state)
                        print(f"Action {action} invalid for Belief {idx + 1}, keeping state")
                        continue
                    new_state = [row[:] for row in state]
                    new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
                    new_belief.append(new_state)
                    print(f"Applied action {action} to Belief {idx + 1}")
                new_belief_tuple = tuple(tuple(tuple(row) for row in state) for state in new_belief)
                if new_belief_tuple not in visited:
                    visited.add(new_belief_tuple)
                    print(f"New belief state added, size: {len(new_belief)}")
                    if all(state == GOAL_STATE for state in new_belief):
                        print("Solution found!")
                        return path + [new_belief]
                    queue.append((new_belief, path + [new_belief]))

        print("No solution found!")
        self.puzzle.error_message = "Không tìm thấy giải pháp!"
        return None

    def solve(self):
        start_time = time.time()
        path = self.sensorless_bfs()
        end_time = time.time()
        self.puzzle.execution_time = end_time - start_time
        return path


def draw_gradient_background(screen):
    gradient_surface = pygame.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(BG_GRADIENT_TOP[0] * (1 - ratio) + BG_GRADIENT_BOTTOM[0] * ratio)
        g = int(BG_GRADIENT_TOP[1] * (1 - ratio) + BG_GRADIENT_BOTTOM[1] * ratio)
        b = int(BG_GRADIENT_TOP[2] * (1 - ratio) + BG_GRADIENT_BOTTOM[2] * ratio)
        pygame.draw.line(gradient_surface, (r, g, b), (0, y), (WIDTH, y))
    screen.blit(gradient_surface, (0, 0))


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


def draw_ui(screen, font, puzzle, is_running, is_paused):
    shadow_rect = pygame.Rect(CONTROL_PANEL_X + 4, CONTROL_PANEL_Y + 4, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT)
    pygame.draw.rect(screen, (30, 30, 30, 100), shadow_rect, border_radius=20)

    panel_rect = pygame.Rect(CONTROL_PANEL_X, CONTROL_PANEL_Y, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=20)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect, 3, border_radius=20)

    header_rect = pygame.Rect(CONTROL_PANEL_X, CONTROL_PANEL_Y, CONTROL_PANEL_WIDTH, 50)
    pygame.draw.rect(screen, BLUE, header_rect, border_radius=20)
    pygame.draw.rect(screen, DARK_GRAY, header_rect, 3, border_radius=20)
    header_bottom_rect = pygame.Rect(CONTROL_PANEL_X, CONTROL_PANEL_Y + 25, CONTROL_PANEL_WIDTH, 25)
    pygame.draw.rect(screen, BLUE, header_bottom_rect)

    title_text = TITLE_FONT.render("Bảng Điều Khiển", True, WHITE)
    title_rect = title_text.get_rect(center=(CONTROL_PANEL_X + CONTROL_PANEL_WIDTH // 2, CONTROL_PANEL_Y + 25))
    screen.blit(title_text, title_rect)

    status_y = CONTROL_PANEL_Y + 65
    pygame.draw.circle(screen, DARK_GREEN, (CONTROL_PANEL_X + 25, status_y), 12)
    step_icon = SMALL_FONT.render("⟳", True, WHITE)
    step_rect = step_icon.get_rect(center=(CONTROL_PANEL_X + 25, status_y))
    screen.blit(step_icon, step_rect)

    steps_text = font.render(f"Số bước: {puzzle.move_count}", True, BLACK)
    screen.blit(steps_text, (CONTROL_PANEL_X + 45, status_y - 12))

    pygame.draw.circle(screen, DARK_YELLOW, (CONTROL_PANEL_X + 25, status_y + 40), 12)
    time_icon = SMALL_FONT.render("⏱", True, WHITE)
    time_rect = time_icon.get_rect(center=(CONTROL_PANEL_X + 25, status_y + 40))
    screen.blit(time_icon, time_rect)

    time_text = font.render(f"Thời gian: {puzzle.execution_time:.2f}s", True, BLACK)
    screen.blit(time_text, (CONTROL_PANEL_X + 45, status_y + 28))

    button_spacing = 55
    button_y = status_y + 85
    button_x = CONTROL_PANEL_X + (CONTROL_PANEL_WIDTH - BUTTON_WIDTH) // 2

    start_rect = pygame.Rect(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 2, border_radius=10)
    start_text = font.render("Bắt đầu", True, WHITE if not is_running else DARK_GRAY)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    pause_rect = pygame.Rect(button_x, button_y + button_spacing, BUTTON_WIDTH, BUTTON_HEIGHT)
    if is_paused:
        draw_gradient_rect(screen, pause_rect, DARK_BLUE, LIGHT_BLUE)
        pause_text = font.render("Tiếp tục", True, WHITE)
    else:
        draw_gradient_rect(screen, pause_rect, DARK_YELLOW, LIGHT_YELLOW)
        pause_text = font.render("Tạm dừng", True, WHITE)
    pygame.draw.rect(screen, DARK_GRAY, pause_rect, 2, border_radius=10)
    pause_text_rect = pause_text.get_rect(center=pause_rect.center)
    screen.blit(pause_text, pause_text_rect)

    reset_rect = pygame.Rect(button_x, button_y + 2 * button_spacing, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 2, border_radius=10)
    reset_text = font.render("Đặt lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    return start_rect, pause_rect, reset_rect


def find_highlight_positions(puzzle, path, step):
    if step + 1 >= len(path):
        return {}
    current_belief = path[step]
    next_belief = path[step + 1]
    highlight_positions = {}
    for idx, (current_state, next_state) in enumerate(zip(current_belief, next_belief)):
        if current_state == GOAL_STATE:
            print(f"Belief {idx + 1} at goal, skipping highlights")
            continue
        empty_i, empty_j = puzzle.find_empty_in_state(current_state)
        if empty_i is None or empty_j is None:
            print(f"Error: No empty tile in Belief {idx + 1}")
            continue
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = empty_i + di, empty_j + dj
            if 0 <= ni < 3 and 0 <= nj < 3 and next_state[empty_i][empty_j] == current_state[ni][nj]:
                highlight_positions[idx] = [(ni, nj)]
                puzzle.start_animation(idx, current_state, next_state)
                break
    return highlight_positions


def print_belief_state(belief_states, step):
    print(f"\nBước {step}:")
    for idx, state in enumerate(belief_states):
        status = "Hoàn thành" if state == GOAL_STATE else "Đang xử lý"
        print(f"Niềm tin {idx + 1} ({status}):")
        for row in state:
            print(row)


def main():
    initial_belief = [BELIEF_STATE_1, BELIEF_STATE_2, BELIEF_STATE_3]
    print("Initial belief states:")
    for idx, state in enumerate(initial_belief):
        print(f"Niềm tin {idx + 1}: {'Matches GOAL' if state == GOAL_STATE else 'Unique'}")
        for row in state:
            print(row)
    puzzle = Puzzle(initial_belief)
    running = True
    solving = False
    paused = False
    path = []
    step = 0
    animation_time = 0
    clock = pygame.time.Clock()

    try:
        while running:
            dt = clock.tick(60) / 1000.0
            animation_time += dt
            puzzle.update_animations(dt)
            draw_gradient_background(screen)

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    start_rect, pause_rect, reset_rect = draw_ui(screen, FONT, puzzle, solving, paused)
                    if start_rect.collidepoint(x, y) and not solving:
                        print("Start button clicked")
                        solving = True
                        paused = False
                        puzzle.error_message = ""
                        solver = PuzzleSolver(puzzle)
                        path = solver.solve()
                        step = 0
                        puzzle.move_count = 0
                        if not path:
                            solving = False
                            print("Không tìm thấy giải pháp!")
                        else:
                            print("Đã tìm thấy giải pháp!")
                    if pause_rect.collidepoint(x, y) and solving:
                        print("Pause/Continue button clicked")
                        paused = not paused
                    if reset_rect.collidepoint(x, y):
                        print("Reset button clicked")
                        puzzle = Puzzle(initial_belief)
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        puzzle.error_message = ""
                        solving = False
                        paused = False
                        path = []
                        step = 0
                        animation_time = 0
                        print("Đã đặt lại trạng thái ban đầu.")

            if solving and not paused and path:
                if step < len(path):
                    if all(len(anim) == 0 for anim in puzzle.animations):
                        print(f"Rendering step {step}")
                        highlight_positions = find_highlight_positions(puzzle, path, step)
                        puzzle.belief_states = [state[:] for state in path[step]]
                        puzzle.move_count = step
                        print_belief_state(puzzle.belief_states, step)
                        step += 1
                        pygame.time.wait(1000)
                        animation_time = 0
                    puzzle.draw(screen, highlight_positions, animation_time=animation_time)
                else:
                    puzzle.draw(screen, show_goal_only=True, animation_time=animation_time)
            else:
                puzzle.draw(screen, animation_time=animation_time)

            draw_ui(screen, FONT, puzzle, solving, paused)
            pygame.display.flip()

    except Exception as e:
        print(f"An error occurred: {e}")
        puzzle.error_message = f"Lỗi: {str(e)}"
        screen.fill((255, 200, 200))
        error_text = FONT.render(puzzle.error_message, True, RED)
        error_rect = error_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(error_text, error_rect)
        pygame.display.flip()
        pygame.time.wait(2000)

    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
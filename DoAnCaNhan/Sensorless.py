import pygame
import sys
from pygame.locals import *
import copy
from collections import deque
import time

# Khởi tạo Pygame
pygame.init()

# Cài đặt màn hình
WIDTH = 1280
HEIGHT = 720
CELL_SIZE = 100
GRID_SIZE = 3
BUTTON_WIDTH = 300
BUTTON_HEIGHT = 60
CONTROL_PANEL_WIDTH = 700
CONTROL_PANEL_X = 540
screen = pygame.display.set_mode((WIDTH + 30, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver - Sensorless BFS")

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
GOAL_COLOR = (144, 238, 144)  # Màu xanh lá nhạt cho đích đến

# Font hỗ trợ tiếng Việt
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 36)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 48)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 50)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 36)
    TITLE_FONT = pygame.font.Font(None, 48)
    NUMBER_FONT = pygame.font.Font(None, 50)

# Trạng thái mục tiêu
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Hai trạng thái niềm tin ban đầu
BELIEF_STATE_1 = [
    [1, 2, 3],
    [0, 5, 6],
    [4, 7, 8]
]
BELIEF_STATE_2 = [
    [1, 0, 3],
    [4, 2, 6],
    [7, 5, 8]
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

    def draw(self, screen, highlight_pos=None, show_goal_only=False):
        if show_goal_only and all(state == GOAL_STATE for state in self.belief_states):
            # Hiển thị duy nhất GOAL_STATE sát bên trái
            x_offset = 50  # Đặt sát bên trái như các lưới niềm tin
            y_offset = HEIGHT // 2 - CELL_SIZE * 1.5  # Căn giữa theo chiều dọc
            label_text = TITLE_FONT.render("Đích đến", True, BLUE)
            label_rect = label_text.get_rect(center=(x_offset + CELL_SIZE * 1.5, y_offset - 30))
            screen.blit(label_text, label_rect)
            for i in range(3):
                for j in range(3):
                    x0 = j * CELL_SIZE + x_offset
                    y0 = i * CELL_SIZE + y_offset
                    rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, GOAL_COLOR, rect, border_radius=10)
                    pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                    if GOAL_STATE[i][j] != 0:
                        text = NUMBER_FONT.render(str(GOAL_STATE[i][j]), True, WHITE)
                        text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                        screen.blit(text, text_rect)
        else:
            for idx, state in enumerate(self.belief_states):
                y_offset = idx * (CELL_SIZE * 3 + 50) + 50
                x_offset = 50
                label_text = TITLE_FONT.render(f"Niềm tin {idx + 1}", True, BLUE)
                label_rect = label_text.get_rect(center=(x_offset + CELL_SIZE * 1.5, y_offset - 30))
                screen.blit(label_text, label_rect)
                for i in range(3):
                    for j in range(3):
                        x0 = j * CELL_SIZE + x_offset
                        y0 = i * CELL_SIZE + y_offset
                        rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                        if state[i][j] == 0:
                            pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                        elif highlight_pos and (i, j) == highlight_pos:
                            pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=10)
                        else:
                            pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                        pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                        if state[i][j] != 0:
                            text = NUMBER_FONT.render(str(state[i][j]), True, WHITE)
                            text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                            screen.blit(text, text_rect)

    def find_empty_in_state(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None


class PuzzleSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle

    def sensorless_bfs(self):
        print("Kiểm tra trạng thái ban đầu:", self.puzzle.belief_states)
        if all(state == GOAL_STATE for state in self.puzzle.belief_states):
            print("Cả hai trạng thái đã ở GOAL_STATE!")
            return [self.puzzle.belief_states]

        initial_belief = tuple(tuple(tuple(row) for row in state) for state in self.puzzle.belief_states)
        queue = deque([(self.puzzle.belief_states, [self.puzzle.belief_states])])
        visited = {initial_belief}
        max_iterations = 100000
        iteration = 0

        while queue:
            if iteration >= max_iterations:
                print("Đã vượt quá số lần lặp tối đa. Có thể không tồn tại giải pháp.")
                return None
            iteration += 1

            current_belief, path = queue.popleft()
            print(f"Iteration {iteration}, Current belief: {current_belief}")

            actions = set()
            for state in current_belief:
                empty_i, empty_j = self.puzzle.find_empty_in_state(state)
                if empty_i is None or empty_j is None:
                    print("Lỗi: Không tìm thấy ô trống trong trạng thái!", state)
                    return None
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = empty_i + di, empty_j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        actions.add((di, dj))

            print(f"Possible actions: {actions}")
            if not actions:
                print("Không có hành động nào khả thi cho trạng thái hiện tại!")
                continue

            for action in actions:
                new_belief = []
                for state in current_belief:
                    empty_i, empty_j = self.puzzle.find_empty_in_state(state)
                    ni, nj = empty_i + action[0], empty_j + action[1]
                    if not (0 <= ni < 3 and 0 <= nj < 3):
                        continue
                    new_state = [row[:] for row in state]
                    new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
                    new_belief.append(new_state)

                new_belief_tuple = tuple(tuple(tuple(row) for row in state) for state in new_belief)
                print(f"New belief: {new_belief}")
                if new_belief_tuple not in visited:
                    visited.add(new_belief_tuple)
                    if all(state == GOAL_STATE for state in new_belief):
                        print("Đã tìm thấy giải pháp!")
                        return path + [new_belief]
                    queue.append((new_belief, path + [new_belief]))

        print("Không tìm thấy giải pháp sau khi duyệt hết!")
        return None

    def solve(self):
        start_time = time.time()
        print("Starting Sensorless BFS...")
        path = self.sensorless_bfs()
        end_time = time.time()
        self.puzzle.execution_time = end_time - start_time
        if path:
            print(f"Sensorless BFS completed in {self.puzzle.execution_time:.2f}s")
        else:
            print("Không tìm thấy giải pháp!")
        return path


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
    title_text = TITLE_FONT.render("Sensorless BFS", True, BLUE)
    screen.blit(title_text, (CONTROL_PANEL_X + 150, 20))
    shadow_rect = pygame.Rect(CONTROL_PANEL_X + 5, 85, CONTROL_PANEL_WIDTH, 580)
    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=20)
    pygame.draw.rect(screen, LIGHT_BLUE, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 580), border_radius=20)
    pygame.draw.rect(screen, DARK_GRAY, (CONTROL_PANEL_X, 80, CONTROL_PANEL_WIDTH, 580), 5, border_radius=20)

    steps_text = font.render(f"Số bước thực hiện: {puzzle.move_count}", True, BLACK)
    steps_rect = steps_text.get_rect(topleft=(CONTROL_PANEL_X + 30, 150))
    screen.blit(steps_text, steps_rect)

    time_text = font.render(f"Thời gian: {puzzle.execution_time:.2f}s", True, BLACK)
    time_rect = time_text.get_rect(topleft=(CONTROL_PANEL_X + 30, 200))
    screen.blit(time_text, time_rect)

    button_x = CONTROL_PANEL_X + (CONTROL_PANEL_WIDTH - BUTTON_WIDTH) // 2
    start_rect = pygame.Rect(button_x, 360, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=15)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 5, border_radius=15)
    start_text = font.render("Bắt đầu", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    pause_rect = pygame.Rect(button_x, 460, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_paused:
        draw_gradient_rect(screen, pause_rect, DARK_YELLOW, LIGHT_YELLOW)
    else:
        pygame.draw.rect(screen, BLUE, pause_rect, border_radius=15)
    pygame.draw.rect(screen, DARK_GRAY, pause_rect, 5, border_radius=15)
    pause_text = font.render("Tạm Dừng" if not is_paused else "Tiếp Tục", True, WHITE)
    pause_text_rect = pause_text.get_rect(center=pause_rect.center)
    screen.blit(pause_text, pause_text_rect)

    reset_rect = pygame.Rect(button_x, 560, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 5, border_radius=15)
    reset_text = font.render("Đặt Lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    return start_rect, pause_rect, reset_rect


def find_highlight_pos(puzzle, path, step):
    if step + 1 >= len(path):
        return None, None
    current_belief = path[step]
    next_belief = path[step + 1]
    empty_pos = None
    next_pos = None
    for idx, state in enumerate(current_belief):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    empty_pos = (i, j)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 3 and 0 <= nj < 3 and next_belief[idx][i][j] == current_belief[idx][ni][nj]:
                            next_pos = (ni, nj)
                            return empty_pos, next_pos
    return empty_pos, next_pos


def print_belief_state(belief_states, step):
    print(f"\nBước {step}:")
    for idx, state in enumerate(belief_states):
        print(f"Niềm tin {idx + 1}:")
        for row in state:
            print(row)


def main():
    initial_belief = [BELIEF_STATE_1, BELIEF_STATE_2]
    puzzle = Puzzle(initial_belief)
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
                    start_rect, pause_rect, reset_rect = draw_ui(screen, FONT, puzzle, solving, paused)
                    if start_rect.collidepoint(x, y) and not solving:
                        solving = True
                        paused = False
                        solver = PuzzleSolver(puzzle)
                        path = solver.solve()
                        step = 0
                        puzzle.move_count = 0
                        if not path:
                            solving = False
                            print("Không thể tiếp tục vì không tìm thấy giải pháp!")
                        else:
                            print("Đã tìm thấy giải pháp! Bắt đầu in các trạng thái:")
                    if pause_rect.collidepoint(x, y) and solving:
                        paused = not paused
                    if reset_rect.collidepoint(x, y):
                        puzzle = Puzzle(initial_belief)
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        solving = False
                        paused = False
                        path = []
                        step = 0
                        print("Đã đặt lại trạng thái ban đầu:")
            if solving and not paused and path:
                if step < len(path):
                    empty_pos, next_pos = find_highlight_pos(puzzle, path, step)
                    puzzle.belief_states = [state[:] for state in path[step]]
                    puzzle.move_count = step
                    print_belief_state(puzzle.belief_states, step)
                    step += 1
                    pygame.time.wait(1000)
                if step >= len(path) or all(state == GOAL_STATE for state in puzzle.belief_states):
                    solving = False
                    print("Đã đạt đích! Hiển thị trạng thái đích đến.")
                    puzzle.belief_states = [GOAL_STATE]  # Thay thế bằng GOAL_STATE duy nhất
            empty_pos, next_pos = find_highlight_pos(puzzle, path, step - 1) if path and step > 0 else (None, None)
            puzzle.draw(screen, next_pos if next_pos and solving else None,
                        step >= len(path) or all(state == GOAL_STATE for state in puzzle.belief_states))
            draw_ui(screen, FONT, puzzle, solving, paused)
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
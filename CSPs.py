import pygame
import sys
from pygame.locals import *
import copy
import time
import random

# Khởi tạo Pygame
pygame.init()

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
pygame.display.set_caption("8-Puzzle Solver Lê Vũ Hải")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (70, 130, 180)
LIGHT_BLUE = (173, 216, 230)
GREEN = (60, 179, 113)
RED = (220, 20, 60)
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (178, 34, 34)
LIGHT_RED = (255, 99, 71)
NEXT_MOVE_COLOR = (255, 182, 193)

# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    FONT = pygame.font.Font(FONT_PATH, 36)
    TITLE_FONT = pygame.font.Font(FONT_PATH, 48)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 60)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    FONT = pygame.font.Font(None, 36)
    TITLE_FONT = pygame.font.Font(None, 48)
    NUMBER_FONT = pygame.font.Font(None, 60)

# Trạng thái ban đầu (trắng) và mục tiêu
INITIAL_STATE = [
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1]
]
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]


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
                if self.state[i][j] == -1:
                    pygame.draw.rect(screen, WHITE, rect, border_radius=10)
                elif self.state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                elif highlight_pos and (i, j) == highlight_pos:
                    pygame.draw.rect(screen, NEXT_MOVE_COLOR, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                if self.state[i][j] not in [-1]:
                    text = NUMBER_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                    screen.blit(text, text_rect)


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


def draw_ui(screen, font, puzzle, is_running, selected_algorithm):
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
    start_rect = pygame.Rect(button_x, 250, BUTTON_WIDTH, BUTTON_HEIGHT)
    if not is_running:
        draw_gradient_rect(screen, start_rect, DARK_GREEN, LIGHT_GREEN)
    else:
        pygame.draw.rect(screen, GRAY, start_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, start_rect, 3, border_radius=10)
    start_text = font.render("Bắt đầu", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_rect.center)
    screen.blit(start_text, start_text_rect)

    algo1_rect = pygame.Rect(button_x, 310, BUTTON_WIDTH, BUTTON_HEIGHT)
    if selected_algorithm == "Backtracking":
        draw_gradient_rect(screen, algo1_rect, BLUE, LIGHT_BLUE)
    else:
        pygame.draw.rect(screen, GRAY, algo1_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, algo1_rect, 3, border_radius=10)
    algo1_text = font.render("Backtracking", True, WHITE)
    algo1_text_rect = algo1_text.get_rect(center=algo1_rect.center)
    screen.blit(algo1_text, algo1_text_rect)

    algo2_rect = pygame.Rect(button_x, 360, BUTTON_WIDTH, BUTTON_HEIGHT)
    if selected_algorithm == "Backtracking Forward":
        draw_gradient_rect(screen, algo2_rect, BLUE, LIGHT_BLUE)
    else:
        pygame.draw.rect(screen, GRAY, algo2_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, algo2_rect, 3, border_radius=10)
    algo2_text = font.render("Backtracking Forward", True, WHITE)
    algo2_text_rect = algo2_text.get_rect(center=algo2_rect.center)
    screen.blit(algo2_text, algo2_text_rect)

    algo3_rect = pygame.Rect(button_x, 410, BUTTON_WIDTH, BUTTON_HEIGHT)
    if selected_algorithm == "Min-Conflicts":
        draw_gradient_rect(screen, algo3_rect, BLUE, LIGHT_BLUE)
    else:
        pygame.draw.rect(screen, GRAY, algo3_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, algo3_rect, 3, border_radius=10)
    algo3_text = font.render("Min-Conflicts", True, WHITE)
    algo3_text_rect = algo3_text.get_rect(center=algo3_rect.center)
    screen.blit(algo3_text, algo3_text_rect)

    reset_rect = pygame.Rect(button_x, 460, BUTTON_WIDTH, BUTTON_HEIGHT)
    draw_gradient_rect(screen, reset_rect, DARK_RED, LIGHT_RED)
    pygame.draw.rect(screen, DARK_GRAY, reset_rect, 3, border_radius=10)
    reset_text = font.render("Đặt Lại", True, WHITE)
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    screen.blit(reset_text, reset_text_rect)

    return start_rect, algo1_rect, algo2_rect, algo3_rect, reset_rect


def delay(ms):
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < ms:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()


class PuzzleSolver:
    def __init__(self, puzzle, screen, algorithm):
        self.puzzle = puzzle
        self.screen = screen
        self.algorithm = algorithm
        self.clock = pygame.time.Clock()

    def forward_check(self, state, used_numbers):
        remaining_numbers = set(range(9)) - used_numbers
        unassigned_positions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == -1]
        required_numbers = {GOAL_STATE[i][j] for i, j in unassigned_positions}
        if len(remaining_numbers) < len(unassigned_positions):
            return False
        for num in required_numbers:
            if num not in remaining_numbers:
                return False
        return True

    def backtracking(self, use_forward_check=False):
        def backtrack(state, next_number, used_numbers, last_pos, depth):
            if next_number > 8:
                if state == GOAL_STATE:
                    print(f"\nĐạt trạng thái mục tiêu tại bước {depth}:")
                    for row in state:
                        print(row)
                    self.puzzle.state = [row[:] for row in state]
                    self.puzzle.move_count = depth
                    self.puzzle.draw(self.screen, None)
                    draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                    pygame.display.flip()
                    delay(1000)
                    return True
                return False

            current_number = next_number if next_number <= 8 else 0
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            cells_to_try = []
            if last_pos:
                for di, dj in directions:
                    ni, nj = last_pos[0] + di, last_pos[1] + dj
                    if 0 <= ni < 3 and 0 <= nj < 3 and state[ni][nj] == -1:
                        cells_to_try.append((ni, nj))
            if not cells_to_try:
                for i in range(3):
                    for j in range(3):
                        if state[i][j] == -1:
                            cells_to_try.append((i, j))

            for i, j in cells_to_try:
                if current_number not in used_numbers:
                    state[i][j] = current_number
                    used_numbers.add(current_number)
                    print(f"\nBước {depth}: Thử gán số {current_number} vào ô ({i}, {j}):")
                    for row in state:
                        print(row)
                    self.puzzle.state = [row[:] for row in state]
                    self.puzzle.move_count = depth
                    self.puzzle.draw(self.screen, (i, j))
                    draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                    pygame.display.flip()
                    delay(1000)

                    if use_forward_check and not self.forward_check(state, used_numbers):
                        print(f"\nForward Checking thất bại tại bước {depth}, quay lui:")
                        for row in state:
                            print(row)
                        state[i][j] = -1
                        used_numbers.remove(current_number)
                        self.puzzle.state = [row[:] for row in state]
                        self.puzzle.move_count = depth
                        self.puzzle.draw(self.screen, None)
                        draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                        pygame.display.flip()
                        delay(1000)
                        continue

                    if backtrack(state, next_number + 1, used_numbers, (i, j), depth + 1):
                        return True

                    state[i][j] = -1
                    used_numbers.remove(current_number)
                    print(f"\nQuay lui (xóa số {current_number} khỏi ô ({i}, {j})):")
                    for row in state:
                        print(row)
                    self.puzzle.state = [row[:] for row in state]
                    self.puzzle.move_count = depth
                    self.puzzle.draw(self.screen, None)
                    draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                    pygame.display.flip()
                    delay(1000)

            return False

        print(f"Bắt đầu {self.algorithm} từ lưới trắng:")
        self.puzzle.move_count = 0
        state = [row[:] for row in self.puzzle.state]
        success = backtrack(state, 1, set(), None, 0)
        if not success:
            print(f"{self.algorithm} không tìm thấy giải pháp")
        return success

    def min_conflicts(self, max_steps=1000):
        # Khởi tạo trạng thái ngẫu nhiên
        numbers = list(range(9))
        random.shuffle(numbers)
        state = [[0 for _ in range(3)] for _ in range(3)]
        idx = 0
        for i in range(3):
            for j in range(3):
                state[i][j] = numbers[idx]
                idx += 1
        self.puzzle.state = [row[:] for row in state]
        self.puzzle.move_count = 0
        print(f"\nTrạng thái ban đầu ngẫu nhiên:")
        for row in state:
            print(row)
        self.puzzle.draw(self.screen, None)
        draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
        pygame.display.flip()
        delay(1000)

        # Hàm đếm xung đột
        def count_conflicts(state):
            conflicts = 0
            for i in range(3):
                for j in range(3):
                    if state[i][j] != GOAL_STATE[i][j]:
                        conflicts += 1
            return conflicts

        # Min-Conflicts
        for step in range(max_steps):
            if state == GOAL_STATE:
                print(f"\nĐạt trạng thái mục tiêu tại bước {step}:")
                for row in state:
                    print(row)
                self.puzzle.state = [row[:] for row in state]
                self.puzzle.move_count = step
                self.puzzle.draw(self.screen, None)
                draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                pygame.display.flip()
                delay(1000)
                return True

            conflicted_cells = [(i, j) for i in range(3) for j in range(3) if state[i][j] != GOAL_STATE[i][j]]
            if not conflicted_cells:
                break

            i, j = random.choice(conflicted_cells)
            min_conflicts = float('inf')
            best_swaps = []
            for ni in range(3):
                for nj in range(3):
                    if (ni, nj) != (i, j):
                        new_state = [row[:] for row in state]
                        new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                        conflicts = count_conflicts(new_state)
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            best_swaps = [(ni, nj)]
                        elif conflicts == min_conflicts:
                            best_swaps.append((ni, nj))

            if best_swaps:
                ni, nj = random.choice(best_swaps)
                state[i][j], state[ni][nj] = state[ni][nj], state[i][j]
                print(f"\nBước {step + 1}: Hoán đổi ô ({i}, {j}) với ô ({ni}, {nj}):")
                for row in state:
                    print(row)
                self.puzzle.state = [row[:] for row in state]
                self.puzzle.move_count = step + 1
                self.puzzle.draw(self.screen, (i, j))
                draw_ui(self.screen, FONT, self.puzzle, True, self.algorithm)
                pygame.display.flip()
                delay(1000)
            else:
                print(f"\nBước {step + 1}: Không tìm thấy hoán đổi tốt, tiếp tục...")
                continue

        print("Min-Conflicts không tìm thấy giải pháp trong số bước tối đa")
        return False

    def solve(self):
        try:
            start_time = time.time()
            print(f"Starting {self.algorithm}...")
            if self.algorithm == "Min-Conflicts":
                success = self.min_conflicts()
            else:
                success = self.backtracking(use_forward_check=(self.algorithm == "Backtracking Forward"))
            end_time = time.time()
            self.puzzle.execution_time = end_time - start_time
            if success:
                print(f"{self.algorithm} completed in {self.puzzle.execution_time:.2f}s")
            return success
        except Exception as e:
            print(f"Lỗi trong {self.algorithm}: {str(e)}")
            return False


def main():
    puzzle = Puzzle([row[:] for row in INITIAL_STATE])
    running = True
    solving = False
    selected_algorithm = "Backtracking"
    clock = pygame.time.Clock()
    try:
        while running:
            screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    start_rect, algo1_rect, algo2_rect, algo3_rect, reset_rect = draw_ui(screen, FONT, puzzle, solving,
                                                                                         selected_algorithm)
                    if start_rect.collidepoint(x, y) and not solving:
                        solving = True
                        solver = PuzzleSolver(puzzle, screen, selected_algorithm)
                        success = solver.solve()
                        solving = False
                        if not success:
                            print("Không tìm thấy giải pháp!")
                    if algo1_rect.collidepoint(x, y) and not solving:
                        selected_algorithm = "Backtracking"
                        print("Đã chọn thuật toán: Backtracking")
                    if algo2_rect.collidepoint(x, y) and not solving:
                        selected_algorithm = "Backtracking Forward"
                        print("Đã chọn thuật toán: Backtracking Forward")
                    if algo3_rect.collidepoint(x, y) and not solving:
                        selected_algorithm = "Min-Conflicts"
                        print("Đã chọn thuật toán: Min-Conflicts")
                    if reset_rect.collidepoint(x, y):
                        puzzle = Puzzle([row[:] for row in INITIAL_STATE])
                        puzzle.move_count = 0
                        puzzle.execution_time = 0
                        solving = False
                        print("Đã đặt lại trạng thái ban đầu:")

            puzzle.draw(screen, None)
            draw_ui(screen, FONT, puzzle, solving, selected_algorithm)
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    except Exception as e:
        print(f"Lỗi chương trình: {str(e)}")
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
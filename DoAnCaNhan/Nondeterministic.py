import pygame
import sys
from pygame.locals import *
import random
import time

# Khởi tạo Pygame
pygame.init()
if not pygame.get_init():
    print("Pygame initialization failed!")
    sys.exit(1)

# Cài đặt màn hình
WIDTH = 400
HEIGHT = 500
CELL_SIZE = 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Nondeterministic")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 215, 0)
GREEN = (60, 179, 113)
RED = (220, 20, 60)

# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    TITLE_FONT = pygame.font.Font(FONT_PATH, 40)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 50)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    TITLE_FONT = pygame.font.Font(None, 40)
    NUMBER_FONT = pygame.font.Font(None, 50)

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

class Puzzle:
    def __init__(self, initial_state):
        self.state = initial_state
        self.last_update_time = 0
        self.update_interval = 2000  # 2 giây
        self.reached_goal = False
        self.failed_action = None  # Vị trí ô khi hành động thất bại
        self.failed_action_time = 0
        self.successful_action = None  # Vị trí ô khi hành động thành công
        self.successful_action_time = 0

    def find_empty(self):
        """Tìm vị trí ô trống."""
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j
        return None

    def is_goal(self):
        """Kiểm tra xem trạng thái hiện tại có phải là trạng thái mục tiêu không."""
        return self.state == GOAL_STATE

    def apply_action(self, action, nondeterministic=True):
        """Thực hiện hành động, có thể thất bại nếu nondeterministic=True."""
        empty_i, empty_j = self.find_empty()
        ni, nj = empty_i, empty_j

        if action == 'Up' and empty_i > 0:
            ni, nj = empty_i - 1, empty_j
        elif action == 'Down' and empty_i < 2:
            ni, nj = empty_i + 1, empty_j
        elif action == 'Left' and empty_j > 0:
            ni, nj = empty_i, empty_j - 1
        elif action == 'Right' and empty_j < 2:
            ni, nj = empty_i, empty_j + 1

        # Mô phỏng nondeterminism: 30% thất bại
        if nondeterministic and random.random() < 0.3:
            self.failed_action = (ni, nj)
            self.failed_action_time = pygame.time.get_ticks()
            self.successful_action = None
            return False  # Hành động thất bại, trạng thái không đổi

        if (ni, nj) != (empty_i, empty_j):
            self.state[empty_i][empty_j], self.state[ni][nj] = self.state[ni][nj], self.state[empty_i][empty_j]
            self.successful_action = (ni, nj)
            self.successful_action_time = pygame.time.get_ticks()
            self.failed_action = None
            return True  # Hành động thành công
        return False  # Hành động không hợp lệ

    def draw(self, screen):
        # Tính toán vị trí để căn giữa bảng 3x3
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height) // 2

        current_time = pygame.time.get_ticks()
        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * CELL_SIZE
                y0 = y_offset + i * CELL_SIZE
                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                if self.reached_goal:
                    pygame.draw.rect(screen, GREEN, rect, border_radius=10)
                elif self.successful_action == (i, j) and current_time - self.successful_action_time < 500:
                    pygame.draw.rect(screen, YELLOW, rect, border_radius=10)
                elif self.failed_action == (i, j) and current_time - self.failed_action_time < 500:
                    pygame.draw.rect(screen, RED, rect, border_radius=10)
                elif self.state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                if self.state[i][j] != 0:
                    text = NUMBER_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                    screen.blit(text, text_rect)

def and_or_search(puzzle):
    """Thuật toán AND-OR search cho môi trường nondeterministic."""
    def actions(state):
        """Trả về các hành động có thể từ trạng thái."""
        empty_i, empty_j = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    empty_i, empty_j = i, j
                    break
        possible = []
        if empty_i > 0:
            possible.append('Up')
        if empty_i < 2:
            possible.append('Down')
        if empty_j > 0:
            possible.append('Left')
        if empty_j < 2:
            possible.append('Right')
        return possible

    def results(state, action):
        """Trả về tập hợp các trạng thái kết quả sau hành động (nondeterministic)."""
        empty_i, empty_j = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
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

        outcomes = [state]  # Hành động có thể thất bại (giữ nguyên trạng thái)
        if (ni, nj) != (empty_i, empty_j):
            new_state = [row[:] for row in state]
            new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
            outcomes.append(new_state)
        return outcomes

    def is_goal(state):
        """Kiểm tra trạng thái mục tiêu."""
        return state == GOAL_STATE

    def and_or_search_recursive(state, path, visited, depth=0, max_depth=30):
        """Tìm kiếm AND-OR đệ quy."""
        if depth > max_depth:
            return None
        if is_goal(state):
            return path

        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)

        for action in actions(state):
            outcomes = results(state, action)
            plan = []
            all_subplans = []
            for outcome in outcomes:
                subplan = and_or_search_recursive(outcome, path + [action], visited.copy(), depth + 1, max_depth)
                if subplan is None:
                    break
                all_subplans.append(('State', outcome, subplan))
            else:
                plan.append(action)
                if len(outcomes) > 1:
                    plan.append(all_subplans)
                elif all_subplans:
                    plan.extend(all_subplans[0][2])
                if plan:
                    return plan
        return None

    state = puzzle.state
    plan = and_or_search_recursive(state, [], set())
    return plan if plan else ['Right', 'Down', 'Left', 'Up']  # Kế hoạch mặc định nếu thất bại

def execute_plan(puzzle, plan, step):
    """Thực thi kế hoạch, trả về hành động tiếp theo và trạng thái cập nhật."""
    if step >= len(plan):
        return None
    action = plan[step]
    if isinstance(action, str):
        success = puzzle.apply_action(action)
        print(f"Step {step + 1}: Action {action}, Success: {success}")
        return action
    return None

def main():
    print("Starting main loop...")
    puzzle = Puzzle([row[:] for row in INITIAL_STATE])
    plan = and_or_search(puzzle)
    print(f"Generated plan: {plan}")
    running = True
    step = 0
    clock = pygame.time.Clock()
    try:
        while running:
            current_time = pygame.time.get_ticks()
            screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # Cập nhật trạng thái mỗi 2 giây nếu chưa đạt mục tiêu
            if not puzzle.reached_goal and current_time - puzzle.last_update_time >= puzzle.update_interval:
                if step < len(plan):
                    action = execute_plan(puzzle, plan, step)
                    if action:
                        step += 1
                    puzzle.last_update_time = current_time
                if puzzle.is_goal():
                    puzzle.reached_goal = True
                    print("Goal state reached! Stopping updates.")

            # Vẽ tiêu đề
            title_text = TITLE_FONT.render("8-Puzzle Nondeterministic", True, BLACK)
            screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))

            # Vẽ bảng 3x3
            puzzle.draw(screen)

            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
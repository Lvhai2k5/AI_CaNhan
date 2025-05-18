import pygame
import sys
from pygame.locals import *
import random
import time
import math

# Khởi tạo Pygame
pygame.init()
if not pygame.get_init():
    print("Pygame initialization failed!")
    sys.exit(1)

# Cài đặt màn hình
WIDTH = 600
HEIGHT = 750  # Tăng từ 700 lên 750
CELL_SIZE = 120
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Nondeterministic")
icon = pygame.Surface((32, 32))
icon.fill((60, 100, 170))
pygame.display.set_icon(icon)

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BACKGROUND = (245, 245, 245)
BG_PATTERN = (235, 235, 235)
TITLE_COLOR = (30, 30, 80)
SHADOW_COLOR = (0, 0, 0, 40)

# Gradient colors for tiles
GRADIENT_START = (65, 105, 225)  # Royal Blue
GRADIENT_END = (30, 144, 255)  # Dodger Blue
EMPTY_GRADIENT_START = (220, 220, 220)
EMPTY_GRADIENT_END = (200, 200, 200)

# Highlight colors
SUCCESS_COLOR = (46, 204, 113)
FAILURE_COLOR = (231, 76, 60)
HIGHLIGHT_COLOR = (241, 196, 15)
GOAL_COLOR_START = (46, 204, 113)
GOAL_COLOR_END = (39, 174, 96)

# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\segoeui.ttf"  # Segoe UI for modern look
    TITLE_FONT = pygame.font.Font(FONT_PATH, 42)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 52)
    STATUS_FONT = pygame.font.Font(FONT_PATH, 26)
    BUTTON_FONT = pygame.font.Font(FONT_PATH, 22)
    INFO_FONT = pygame.font.Font(FONT_PATH, 16)
except:
    print("Không tìm thấy font Segoe UI. Sử dụng font mặc định.")
    TITLE_FONT = pygame.font.Font(None, 42)
    NUMBER_FONT = pygame.font.Font(None, 52)
    STATUS_FONT = pygame.font.Font(None, 26)
    BUTTON_FONT = pygame.font.Font(None, 22)
    INFO_FONT = pygame.font.Font(None, 16)

# Trạng thái ban đầu (yêu cầu ~10 bước) và mục tiêu
INITIAL_STATE = [
    [8, 7, 6],
    [5, 0, 3],
    [4, 2, 1]
]
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Hiệu ứng âm thanh
try:
    success_sound = pygame.mixer.Sound("success.wav")
    failure_sound = pygame.mixer.Sound("failure.wav")
    win_sound = pygame.mixer.Sound("win.wav")
    has_sound = True
except:
    print("Không tìm thấy file âm thanh. Tắt âm thanh.")
    has_sound = False


# Khởi tạo lớp Button để tạo nút tương tác
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.action = action
        self.is_hovered = False

    def draw(self, surface):
        current_color = self.hover_color if self.is_hovered else self.color
        shadow_rect = self.rect.copy()
        shadow_rect.y += 4
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        shadow_surface.fill((0, 0, 0, 50))
        surface.blit(shadow_surface, shadow_rect)
        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)
        pygame.draw.rect(surface, DARK_GRAY, self.rect, 2, border_radius=8)
        text_surf = BUTTON_FONT.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, pos):
        if self.rect.collidepoint(pos) and self.action:
            return self.action()
        return False


# Tạo hiệu ứng particle
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = random.randint(4, 8)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)
        self.lifetime = random.randint(20, 60)

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def add_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def update(self):
        for particle in self.particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                self.particles.remove(particle)

    def draw(self, surface):
        for particle in self.particles:
            particle.draw(surface)


class Puzzle:
    def __init__(self, initial_state):
        self.state = [row[:] for row in initial_state]
        self.last_update_time = 0
        self.update_interval = 2000  # 2 giây
        self.reached_goal = False
        self.goal_animation_start = 0
        self.successful_action = None  # [ô di chuyển, ô trống]
        self.successful_action_time = 0
        self.failed_action = None  # [ô được chọn, ô trống]
        self.failed_action_time = 0
        self.current_action = None
        self.action_status = None
        self.particles = ParticleSystem()
        self.tile_rotations = {(i, j): 0 for i in range(3) for j in range(3)}
        self.tile_scales = {(i, j): 1.0 for i in range(3) for j in range(3)}
        self.moves_count = 0
        self.start_time = pygame.time.get_ticks()
        self.auto_solving = False
        self.scale_animation = {}
        self.last_empty = self.find_empty()

    def get_elapsed_time(self):
        if self.reached_goal:
            return (self.goal_animation_start - self.start_time) / 1000
        return (pygame.time.get_ticks() - self.start_time) / 1000

    def find_empty(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j
        return None, None

    def is_goal(self):
        return self.state == GOAL_STATE

    def apply_action(self, action, nondeterministic=True):
        empty_i, empty_j = self.find_empty()
        if empty_i is None or empty_j is None:
            return False
        ni, nj = empty_i, empty_j

        if action == 'Up' and empty_i > 0:
            ni, nj = empty_i - 1, empty_j
        elif action == 'Down' and empty_i < 2:
            ni, nj = empty_i + 1, empty_j
        elif action == 'Left' and empty_j > 0:
            ni, nj = empty_i, empty_j - 1
        elif action == 'Right' and empty_j < 2:
            ni, nj = empty_i, empty_j + 1

        self.current_action = action

        if (ni, nj) == (empty_i, empty_j):
            return False

        if nondeterministic and random.random() < 0.3:
            self.failed_action = [(ni, nj), (empty_i, empty_j)]
            self.failed_action_time = pygame.time.get_ticks()
            self.successful_action = None
            self.action_status = "Thất bại"
            self.shake_tile(ni, nj)
            x, y = self.get_tile_center(ni, nj)
            self.particles.add_particles(x, y, 20, FAILURE_COLOR)
            if has_sound:
                failure_sound.play()
            return False

        if (ni, nj) != (empty_i, empty_j):
            self.last_empty = (empty_i, empty_j)
            self.state[empty_i][empty_j], self.state[ni][nj] = self.state[ni][nj], self.state[empty_i][empty_j]
            self.successful_action = [(ni, nj), (empty_i, empty_j)]
            self.successful_action_time = pygame.time.get_ticks()
            self.failed_action = None
            self.action_status = "Thành công"
            self.moves_count += 1
            self.scale_tile(empty_i, empty_j)
            x, y = self.get_tile_center(empty_i, empty_j)
            self.particles.add_particles(x, y, 15, SUCCESS_COLOR)
            if has_sound:
                success_sound.play()
            if self.is_goal() and not self.reached_goal:
                self.reached_goal = True
                self.goal_animation_start = pygame.time.get_ticks()
                for i in range(3):
                    for j in range(3):
                        x, y = self.get_tile_center(i, j)
                        self.particles.add_particles(x, y, 10, SUCCESS_COLOR)
                if has_sound:
                    win_sound.play()
            return True
        return False

    def get_tile_center(self, i, j):
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height - 100) // 2 + 80
        x = x_offset + j * CELL_SIZE + CELL_SIZE // 2
        y = y_offset + i * CELL_SIZE + CELL_SIZE // 2
        return x, y

    def shake_tile(self, i, j):
        self.tile_rotations[(i, j)] = random.uniform(-5, 5)

    def scale_tile(self, i, j):
        self.scale_animation[(i, j)] = {
            'start_time': pygame.time.get_ticks(),
            'duration': 300
        }

    def draw_tile(self, screen, i, j, x0, y0, value, special_effect=None):
        current_time = pygame.time.get_ticks()
        scale = 1.0
        rotation = 0
        if (i, j) in self.scale_animation:
            anim = self.scale_animation[(i, j)]
            elapsed = current_time - anim['start_time']
            if elapsed < anim['duration']:
                progress = elapsed / anim['duration']
                scale = 1.0 + 0.2 * math.sin(progress * math.pi)
            else:
                self.scale_animation.pop((i, j))
        if self.failed_action and (i, j) in self.failed_action and current_time - self.failed_action_time < 500:
            elapsed = current_time - self.failed_action_time
            rotation = 5 * math.sin(elapsed * 0.05)
        size = int(CELL_SIZE * scale)
        offset = (size - CELL_SIZE) // 2
        tile_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        if value == 0:
            if special_effect == "goal":
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        EMPTY_GRADIENT_START, EMPTY_GRADIENT_END, border_radius=15)
            else:
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        EMPTY_GRADIENT_START, EMPTY_GRADIENT_END, border_radius=15)
        else:
            if special_effect == "goal":
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        GOAL_COLOR_START, GOAL_COLOR_END, border_radius=15)
            elif special_effect == "success":
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        SUCCESS_COLOR,
                                        (SUCCESS_COLOR[0] - 20, SUCCESS_COLOR[1] - 20, SUCCESS_COLOR[2] - 20),
                                        border_radius=15)
            elif special_effect == "failure":
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        FAILURE_COLOR,
                                        (FAILURE_COLOR[0] - 20, FAILURE_COLOR[1] - 20, FAILURE_COLOR[2] - 20),
                                        border_radius=15)
            else:
                self.draw_gradient_rect(tile_surface, pygame.Rect(0, 0, size, size),
                                        GRADIENT_START, GRADIENT_END, border_radius=15)
            text = NUMBER_FONT.render(str(value), True, WHITE)
            text_rect = text.get_rect(center=(size // 2, size // 2))
            shadow_text = NUMBER_FONT.render(str(value), True, (0, 0, 0, 100))
            shadow_rect = shadow_text.get_rect(center=(text_rect.center[0] + 2, text_rect.center[1] + 2))
            tile_surface.blit(shadow_text, shadow_rect)
            tile_surface.blit(text, text_rect)
        if rotation != 0:
            tile_surface = pygame.transform.rotate(tile_surface, rotation)
        tile_rect = tile_surface.get_rect()
        tile_rect.center = (x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2)
        screen.blit(tile_surface, tile_rect)

    def draw_gradient_rect(self, surface, rect, color1, color2, border_radius=0):
        """Vẽ hình chữ nhật với hiệu ứng gradient từ color1 đến color2"""
        temp_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        h = rect.height
        for i in range(h):
            ratio = i / h if h > 0 else 0
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            pygame.draw.rect(temp_surface, (r, g, b), (0, i, rect.width, 1))
        if border_radius > 0:
            mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, rect.width, rect.height), border_radius=border_radius)
            temp_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
        surface.blit(temp_surface, (rect.x, rect.y))

    def draw_board_background(self, screen, x_offset, y_offset, grid_width, grid_height):
        board_rect = pygame.Rect(x_offset - 10, y_offset - 10, grid_width + 20, grid_height + 20)
        shadow_rect = board_rect.copy()
        shadow_rect.x += 5
        shadow_rect.y += 5
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        shadow_surface.fill((0, 0, 0, 50))
        screen.blit(shadow_surface, shadow_rect)
        pygame.draw.rect(screen, WHITE, board_rect, border_radius=15)
        pygame.draw.rect(screen, DARK_GRAY, board_rect, 2, border_radius=15)
        for i in range(4):
            pygame.draw.line(screen, BG_PATTERN,
                             (board_rect.x, board_rect.y + i * grid_height // 3),
                             (board_rect.x + board_rect.width, board_rect.y + i * grid_height // 3), 1)
            pygame.draw.line(screen, BG_PATTERN,
                             (board_rect.x + i * grid_width // 3, board_rect.y),
                             (board_rect.x + i * grid_width // 3, board_rect.y + board_rect.height), 1)

    def draw(self, screen):
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height - 100) // 2 + 80
        current_time = pygame.time.get_ticks()
        self.draw_board_background(screen, x_offset, y_offset, grid_width, grid_height)
        self.particles.update()
        self.particles.draw(screen)
        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * CELL_SIZE
                y0 = y_offset + i * CELL_SIZE
                special_effect = None
                if self.reached_goal:
                    special_effect = "goal"
                elif self.successful_action and (
                i, j) in self.successful_action and current_time - self.successful_action_time < 500:
                    special_effect = "success"
                elif self.failed_action and (
                i, j) in self.failed_action and current_time - self.failed_action_time < 500:
                    special_effect = "failure"
                self.draw_tile(screen, i, j, x0, y0, self.state[i][j], special_effect)
        # Di chuyển status_text (Hành động) lên trên lưới
        if self.current_action and current_time - max(self.successful_action_time or 0,
                                                      self.failed_action_time or 0) < 750:  # Giảm từ 1000ms xuống 750ms
            status_color = SUCCESS_COLOR if self.action_status == "Thành công" else FAILURE_COLOR
            status_text = STATUS_FONT.render(f"Hành động: {self.current_action} - {self.action_status}", True,
                                             status_color)
            status_rect = status_text.get_rect(center=(WIDTH // 2, y_offset - 30))  # Đặt trên lưới
            shadow_text = STATUS_FONT.render(f"Hành động: {self.current_action} - {self.action_status}", True,
                                             SHADOW_COLOR)
            shadow_rect = shadow_text.get_rect(center=(status_rect.center[0] + 2, status_rect.center[1] + 2))
            screen.blit(shadow_text, shadow_rect)
            screen.blit(status_text, status_rect)
        # Giữ info_text (Số bước và Thời gian) bên dưới lưới
        info_text = INFO_FONT.render(f"Số bước: {self.moves_count} | Thời gian: {self.get_elapsed_time():.1f}s", True,
                                     DARK_GRAY)
        info_rect = info_text.get_rect(center=(WIDTH // 2, y_offset + grid_height + 60))
        screen.blit(info_text, info_rect)
        if self.reached_goal:
            elapsed = current_time - self.goal_animation_start
            scale = 1.0 + 0.1 * math.sin(elapsed * 0.005)
            goal_text = TITLE_FONT.render("HOÀN THÀNH!", True, SUCCESS_COLOR)
            goal_text = pygame.transform.rotozoom(goal_text, 0, scale)
            goal_rect = goal_text.get_rect(center=(WIDTH // 2, y_offset - 60))  # Điều chỉnh để không đè lên status_text
            shadow_goal = TITLE_FONT.render("HOÀN THÀNH!", True, SHADOW_COLOR)
            shadow_goal = pygame.transform.rotozoom(shadow_goal, 0, scale)
            shadow_rect = shadow_goal.get_rect(center=(goal_rect.center[0] + 3, goal_rect.center[1] + 3))
            screen.blit(shadow_goal, shadow_rect)
            screen.blit(goal_text, goal_rect)


def and_or_search(puzzle):
    def actions(state):
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
        outcomes = [state]
        if (ni, nj) != (empty_i, empty_j):
            new_state = [row[:] for row in state]
            new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
            outcomes.append(new_state)
        return outcomes

    def is_goal(state):
        return state == GOAL_STATE

    def and_or_search_recursive(state, path, visited, depth=0, max_depth=30):
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
    return plan if plan else [
        'Up', 'Right', 'Down', 'Left', 'Up', 'Right', 'Down', 'Left', 'Up', 'Right', 'Up'
    ]


def execute_plan(puzzle, plan, step):
    if step >= len(plan):
        return None
    action = plan[step]
    if isinstance(action, str):
        success = puzzle.apply_action(action)
        print(f"Step {step + 1}: Action {action}, Success: {success}")
        return action
    return None


def reset_puzzle():
    return Puzzle([row[:] for row in INITIAL_STATE])


def toggle_auto_solve(puzzle):
    puzzle.auto_solving = not puzzle.auto_solving
    if puzzle.auto_solving:
        puzzle.start_time = pygame.time.get_ticks()
        return and_or_search(puzzle)
    return puzzle.plan if hasattr(puzzle, 'plan') else and_or_search(puzzle)


def draw_background(screen):
    screen.fill(BACKGROUND)
    for i in range(0, WIDTH, 20):
        pygame.draw.line(screen, BG_PATTERN, (i, 0), (i, HEIGHT), 1)
    for i in range(0, HEIGHT, 20):
        pygame.draw.line(screen, BG_PATTERN, (0, i), (WIDTH, i), 1)


def main():
    print("Starting main loop...")
    puzzle = reset_puzzle()
    plan = and_or_search(puzzle)
    puzzle.plan = plan
    print(f"Generated plan: {plan}")
    running = True
    step = 0
    clock = pygame.time.Clock()

    restart_button = Button(
        WIDTH // 2 - 180, HEIGHT - 80, 150, 50,
        "Khởi động lại",
        (100, 100, 100), (120, 120, 120), WHITE,
        action=reset_puzzle
    )
    auto_solve_button = Button(
        WIDTH // 2 + 30, HEIGHT - 80, 150, 50,
        "Tự động giải",
        (100, 100, 100), (120, 120, 120), WHITE,
        action=lambda: toggle_auto_solve(puzzle)
    )
    buttons = [restart_button, auto_solve_button]

    while running:
        current_time = pygame.time.get_ticks()
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if puzzle.reached_goal:
                    continue
                for button in buttons:
                    result = button.check_click(event.pos)
                    if result and isinstance(result, Puzzle):
                        puzzle = result
                        plan = and_or_search(puzzle)
                        puzzle.plan = plan
                        step = 0
                        print(f"Generated new plan: {plan}")
                    elif result and isinstance(result, list):
                        plan = result
                        step = 0
                        puzzle.plan = plan
                        print(f"Updated plan: {plan}")
                if not puzzle.auto_solving:
                    grid_width = 3 * CELL_SIZE
                    grid_height = 3 * CELL_SIZE
                    x_offset = (WIDTH - grid_width) // 2
                    y_offset = (HEIGHT - grid_height - 100) // 2 + 80
                    click_x, click_y = event.pos
                    if x_offset <= click_x < x_offset + grid_width and y_offset <= click_y < y_offset + grid_height:
                        j = (click_x - x_offset) // CELL_SIZE
                        i = (click_y - y_offset) // CELL_SIZE
                        empty_i, empty_j = puzzle.find_empty()
                        if empty_i is not None and empty_j is not None:
                            if abs(i - empty_i) + abs(j - empty_j) == 1:
                                if i < empty_i:
                                    puzzle.apply_action('Down')
                                elif i > empty_i:
                                    puzzle.apply_action('Up')
                                elif j < empty_j:
                                    puzzle.apply_action('Right')
                                elif j > empty_j:
                                    puzzle.apply_action('Left')

        if puzzle.auto_solving and not puzzle.reached_goal and current_time - puzzle.last_update_time >= puzzle.update_interval:
            if step < len(plan):
                action = execute_plan(puzzle, plan, step)
                if action:
                    step += 1
                puzzle.last_update_time = current_time
            if puzzle.is_goal():
                puzzle.reached_goal = True
                puzzle.auto_solving = False

        draw_background(screen)
        title_text = TITLE_FONT.render("8-Puzzle Nondeterministic", True, TITLE_COLOR)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 50))
        shadow_title = TITLE_FONT.render("8-Puzzle Nondeterministic", True, SHADOW_COLOR)
        shadow_rect = shadow_title.get_rect(center=(title_rect.center[0] + 3, title_rect.center[1] + 3))
        screen.blit(shadow_title, shadow_rect)
        screen.blit(title_text, title_rect)
        puzzle.draw(screen)
        for button in buttons:
            button.update(mouse_pos)
            button.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
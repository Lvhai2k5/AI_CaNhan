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
WIDTH = 800
HEIGHT = 700
CELL_SIZE = 120
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Partially Observation")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (240, 248, 255)  # Alice Blue
DARK_GRAY = (40, 40, 40)
TILE_COLOR = (70, 130, 180)  # Steel Blue
TILE_HOVER = (100, 149, 237)  # Cornflower Blue
YELLOW = (255, 223, 0)
GREEN = (46, 139, 87)  # Sea Green
RED = (220, 20, 60)  # Crimson


# Gradients và hiệu ứng
def create_gradient(color1, color2, height):
    gradient = pygame.Surface((1, height))
    for y in range(height):
        ratio = y / height
        r = color1[0] * (1 - ratio) + color2[0] * ratio
        g = color1[1] * (1 - ratio) + color2[1] * ratio
        b = color1[2] * (1 - ratio) + color2[2] * ratio
        gradient.set_at((0, y), (r, g, b))
    return gradient


# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\segoeui.ttf"  # Segoe UI for modern look
    TITLE_FONT = pygame.font.Font(FONT_PATH, 60)  # Tăng kích thước từ 48 lên 60
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 56)
    INFO_FONT = pygame.font.Font(FONT_PATH, 22)
    BUTTON_FONT = pygame.font.Font(FONT_PATH, 24)
except:
    print("Không tìm thấy font. Sử dụng font mặc định.")
    TITLE_FONT = pygame.font.Font(None, 60)  # Tăng kích thước từ 48 lên 60
    NUMBER_FONT = pygame.font.Font(None, 56)
    INFO_FONT = pygame.font.Font(None, 22)
    BUTTON_FONT = pygame.font.Font(None, 24)

# Trạng thái mục tiêu
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]


# Tạo hiệu ứng đổ bóng
def draw_tile_with_shadow(surface, color, rect, shadow_size=4, border_radius=15):
    # Vẽ bóng
    shadow_rect = rect.copy()
    shadow_rect.x += shadow_size
    shadow_rect.y += shadow_size
    pygame.draw.rect(surface, (30, 30, 30, 100), shadow_rect, border_radius=border_radius)

    # Vẽ ô chính
    pygame.draw.rect(surface, color, rect, border_radius=border_radius)

    # Vẽ viền
    pygame.draw.rect(surface, DARK_GRAY, rect, 3, border_radius=border_radius)

    # Vẽ hiệu ứng ánh sáng (gradient dọc)
    light_rect = rect.copy()
    light_rect.height = rect.height // 2
    pygame.draw.rect(surface,
                     (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255)),
                     light_rect, border_radius=border_radius)


# Animation constants
ANIMATION_SPEED = 10  # pixels per frame
ANIMATION_DURATION = 300  # milliseconds


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, current_color, self.rect, border_radius=10)
        pygame.draw.rect(surface, DARK_GRAY, self.rect, 2, border_radius=10)

        text_surf = BUTTON_FONT.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class Tile:
    def __init__(self, value, pos, size):
        self.value = value
        self.pos = pos
        self.target_pos = pos
        self.size = size
        self.moving = False
        self.animation_time = 0
        self.animation_start_time = 0
        self.start_pos = pos

    def update(self, current_time):
        if self.moving:
            elapsed = current_time - self.animation_start_time
            if elapsed >= ANIMATION_DURATION:
                self.pos = self.target_pos
                self.moving = False
            else:
                progress = elapsed / ANIMATION_DURATION
                self.pos = (
                    self.start_pos[0] + (self.target_pos[0] - self.start_pos[0]) * progress,
                    self.start_pos[1] + (self.target_pos[1] - self.start_pos[1]) * progress
                )

    def move_to(self, target_pos, current_time):
        self.start_pos = self.pos
        self.target_pos = target_pos
        self.moving = True
        self.animation_start_time = current_time


class Puzzle:
    def __init__(self):
        self.state = self.generate_random_state()
        self.tiles = []
        self.last_update_time = 0
        self.update_interval = 2000  # 2 giây
        self.reached_goal = False
        self.moves = 0
        self.game_time = 0
        self.start_time = pygame.time.get_ticks()
        self.initialize_tiles()

    def initialize_tiles(self):
        self.tiles = []
        # Tính toán vị trí để căn giữa bảng 3x3
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height) // 2 + 50  # Thêm 50px để có không gian cho tiêu đề

        for i in range(3):
            for j in range(3):
                value = self.state[i][j]
                if value != 0:  # Không tạo tile cho ô trống
                    pos = (x_offset + j * CELL_SIZE, y_offset + i * CELL_SIZE)
                    self.tiles.append(Tile(value, pos, CELL_SIZE))

    def generate_random_state(self):
        """Tạo trạng thái ngẫu nhiên với số 1 ở [0][0] và ô trống ở [2][2]."""
        numbers = [2, 3, 4, 5, 6, 7, 8]
        random.shuffle(numbers)
        state = [
            [1, numbers[0], numbers[1]],
            [numbers[2], numbers[3], numbers[4]],
            [numbers[5], numbers[6], 0]
        ]
        return state

    def is_goal(self):
        """Kiểm tra xem trạng thái hiện tại có phải là trạng thái mục tiêu không."""
        return self.state == GOAL_STATE

    def update_tiles(self, current_time):
        for tile in self.tiles:
            tile.update(current_time)

    def reset(self):
        self.state = self.generate_random_state()
        self.reached_goal = False
        self.moves = 0
        self.start_time = pygame.time.get_ticks()
        self.initialize_tiles()

    def draw(self, screen, current_time):
        # Vẽ nền gradient
        gradient = create_gradient((173, 216, 230), (135, 206, 250), HEIGHT)
        gradient = pygame.transform.scale(gradient, (WIDTH, HEIGHT))
        screen.blit(gradient, (0, 0))

        # Vẽ board background
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height) // 2 + 50  # Thêm 50px để có không gian cho tiêu đề

        board_rect = pygame.Rect(x_offset - 20, y_offset - 20,
                                 grid_width + 40, grid_height + 40)
        pygame.draw.rect(screen, (220, 220, 220), board_rect, border_radius=20)
        pygame.draw.rect(screen, DARK_GRAY, board_rect, 3, border_radius=20)

        # Vẽ cells trước
        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * CELL_SIZE
                y0 = y_offset + i * CELL_SIZE
                rect = pygame.Rect(x0, y0, CELL_SIZE - 6, CELL_SIZE - 6)
                pygame.draw.rect(screen, (200, 200, 200), rect, border_radius=15)
                pygame.draw.rect(screen, DARK_GRAY, rect, 2, border_radius=15)

        # Vẽ các tiles
        for tile in self.tiles:
            if tile.value != 0:
                rect = pygame.Rect(tile.pos[0], tile.pos[1], CELL_SIZE - 6, CELL_SIZE - 6)

                # Xác định màu sắc dựa trên giá trị của tile
                if self.reached_goal:
                    tile_color = GREEN
                elif tile.value == 1:
                    tile_color = YELLOW
                else:
                    # Gradient màu giữa các số
                    r = int(70 + (160 * (tile.value / 8)))
                    g = int(130 - (40 * (tile.value / 8)))
                    b = int(180 - (40 * (tile.value / 8)))
                    tile_color = (r, g, b)

                # Vẽ tile với hiệu ứng đổ bóng
                draw_tile_with_shadow(screen, tile_color, rect)

                # Vẽ số
                text = NUMBER_FONT.render(str(tile.value), True, WHITE)
                text_rect = text.get_rect(center=(rect.centerx, rect.centery))

                # Thêm hiệu ứng đổ bóng cho số
                shadow_text = NUMBER_FONT.render(str(tile.value), True, (30, 30, 30))
                shadow_rect = text_rect.copy()
                shadow_rect.x += 2
                shadow_rect.y += 2
                screen.blit(shadow_text, shadow_rect)
                screen.blit(text, text_rect)

        # Vẽ thông báo nếu đã đạt được mục tiêu
        if self.reached_goal:
            congrats_text = TITLE_FONT.render("HOÀN THÀNH!", True, GREEN)
            congrats_rect = congrats_text.get_rect(center=(WIDTH // 2, y_offset - 90))

            # Draw text with shadow
            shadow_text = TITLE_FONT.render("HOÀN THÀNH!", True, DARK_GRAY)
            shadow_rect = congrats_rect.copy()
            shadow_rect.x += 3
            shadow_rect.y += 3
            screen.blit(shadow_text, shadow_rect)
            screen.blit(congrats_text, congrats_rect)


def draw_animated_title(screen, text, x, y, color, shadow_color, current_time):
    # Tạo hiệu ứng rung nhẹ cho tiêu đề
    oscillation = math.sin(current_time / 500) * 3

    # Shadow glow effect (optional)
    glow_radius = 15
    for i in range(glow_radius, 0, -5):
        alpha = int(100 - i * 5)
        if alpha < 0:
            alpha = 0
        glow_color = (shadow_color[0], shadow_color[1], shadow_color[2], alpha)
        glow_surf = TITLE_FONT.render(text, True, glow_color)
        glow_rect = glow_surf.get_rect(center=(x, y + oscillation))
        temp_surf = pygame.Surface(glow_surf.get_size(), pygame.SRCALPHA)
        temp_surf.blit(glow_surf, (0, 0))
        screen.blit(temp_surf, (glow_rect.x - i // 2, glow_rect.y - i // 2))

    # Vẽ đổ bóng cho tiêu đề
    shadow_text = TITLE_FONT.render(text, True, shadow_color)
    shadow_rect = shadow_text.get_rect(center=(x + 4, y + 4 + oscillation))
    screen.blit(shadow_text, shadow_rect)

    # Vẽ tiêu đề
    title_text = TITLE_FONT.render(text, True, color)
    title_rect = title_text.get_rect(center=(x, y + oscillation))
    screen.blit(title_text, title_rect)

    return title_rect.height


def main():
    print("Starting main loop...")
    puzzle = Puzzle()
    running = True
    clock = pygame.time.Clock()
    show_info = False  # Biến để kiểm soát hiển thị hướng dẫn

    # Tạo các buttons
    reset_button = Button(WIDTH // 2 - 150, HEIGHT - 80, 140, 50, "Làm Mới", (50, 120, 190), (70, 140, 210))
    exit_button = Button(WIDTH // 2 + 10, HEIGHT - 80, 140, 50, "Thoát", (190, 50, 50), (210, 70, 70))
    info_button = Button(WIDTH - 600, 20, 400, 40, "Partially Observation", (60, 130, 100), (80, 150, 120))

    try:
        while running:
            current_time = pygame.time.get_ticks()
            screen.fill(BACKGROUND)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                # Xử lý button clicks
                if reset_button.is_clicked(mouse_pos, event):
                    puzzle.reset()
                if exit_button.is_clicked(mouse_pos, event):
                    running = False
                if info_button.is_clicked(mouse_pos, event):
                    show_info = not show_info if 'show_info' in locals() else True

            # Cập nhật hover states của buttons
            reset_button.check_hover(mouse_pos)
            exit_button.check_hover(mouse_pos)
            info_button.check_hover(mouse_pos)

            # Cập nhật trạng thái mới mỗi 2 giây nếu chưa đạt mục tiêu
            if not puzzle.reached_goal and current_time - puzzle.last_update_time >= puzzle.update_interval:
                puzzle.state = puzzle.generate_random_state()
                puzzle.last_update_time = current_time
                puzzle.initialize_tiles()
                print(f"New random state: {puzzle.state}")
                if puzzle.is_goal():
                    puzzle.reached_goal = True
                    print("Goal state reached! Stopping updates.")

            # Cập nhật animation cho các tiles
            puzzle.update_tiles(current_time)

            # Vẽ tiêu đề lớn phía trên khung
            title_height = draw_animated_title(screen, "8-PUZZLE", WIDTH // 2, 80,
                                               (10, 50, 120), (60, 100, 160), current_time)

            # Vẽ subtile
            subtitle_text = TITLE_FONT.render("Partially Observation", True, (40, 80, 140))
            subtitle_shadow = TITLE_FONT.render("Partially Observation", True, (80, 120, 180))

            # Hiệu ứng shadow cho subtitle
            subtitle_rect = subtitle_text.get_rect(center=(WIDTH // 2, 80 + title_height + 10))
            subtitle_shadow_rect = subtitle_shadow.get_rect(center=(WIDTH // 2 + 3, 80 + title_height + 13))

            screen.blit(subtitle_shadow, subtitle_shadow_rect)
            screen.blit(subtitle_text, subtitle_rect)
            screen.blit(subtitle_text, subtitle_rect)

            # Vẽ bảng 3x3
            puzzle.draw(screen, current_time)

            # Vẽ các buttons
            reset_button.draw(screen)
            exit_button.draw(screen)
            info_button.draw(screen)

            # Hiển thị thông tin hướng dẫn nếu được kích hoạt
            if 'show_info' in locals() and show_info:
                info_rect = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 - 150, 400, 300)
                pygame.draw.rect(screen, (240, 240, 255), info_rect, border_radius=15)
                pygame.draw.rect(screen, DARK_GRAY, info_rect, 3, border_radius=15)

                info_title = INFO_FONT.render("HƯỚNG DẪN", True, DARK_GRAY)
                screen.blit(info_title, (info_rect.centerx - info_title.get_width() // 2, info_rect.y + 20))

                info_lines = [
                    "- Game sẽ tự động thay đổi trạng thái mỗi 2 giây",
                    "- Số 1 luôn ở vị trí góc trên bên trái",
                    "- Ô trống luôn ở vị trí góc dưới bên phải",
                    "- Mục tiêu là đạt được trạng thái 1-2-3, 4-5-6, 7-8-0",
                    "",
                    "Nhấn nút [Làm Mới] để bắt đầu lại game",
                    "Nhấn nút [Thoát] để kết thúc game"
                ]

                for i, line in enumerate(info_lines):
                    info_text = INFO_FONT.render(line, True, DARK_GRAY)
                    screen.blit(info_text, (info_rect.x + 20, info_rect.y + 60 + i * 30))

                # Nút đóng hướng dẫn
                close_rect = pygame.Rect(info_rect.right - 40, info_rect.y + 15, 25, 25)
                pygame.draw.rect(screen, RED, close_rect, border_radius=5)
                pygame.draw.line(screen, WHITE, (close_rect.x + 5, close_rect.y + 5),
                                 (close_rect.x + 20, close_rect.y + 20), 2)
                pygame.draw.line(screen, WHITE, (close_rect.x + 20, close_rect.y + 5),
                                 (close_rect.x + 5, close_rect.y + 20), 2)

                # Kiểm tra nếu nhấp vào nút đóng
                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    if close_rect.collidepoint(mouse_pos):
                        show_info = False

            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng (Ctrl+C)")
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
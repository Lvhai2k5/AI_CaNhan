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
WIDTH = 600
HEIGHT = 600
CELL_SIZE = 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Partially Observation")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 215, 0)
GREEN = (60, 179, 113)

# Font
try:
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    TITLE_FONT = pygame.font.Font(FONT_PATH, 40)
    NUMBER_FONT = pygame.font.Font(FONT_PATH, 50)
except:
    print("Không tìm thấy font Arial. Sử dụng font mặc định.")
    TITLE_FONT = pygame.font.Font(None, 40)
    NUMBER_FONT = pygame.font.Font(None, 50)

# Trạng thái mục tiêu
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

class Puzzle:
    def __init__(self):
        self.state = self.generate_random_state()
        self.last_update_time = 0
        self.update_interval = 2000  # 2 giây
        self.reached_goal = False

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

    def draw(self, screen):
        # Tính toán vị trí để căn giữa bảng 3x3
        grid_width = 3 * CELL_SIZE
        grid_height = 3 * CELL_SIZE
        x_offset = (WIDTH - grid_width) // 2
        y_offset = (HEIGHT - grid_height) // 2

        for i in range(3):
            for j in range(3):
                x0 = x_offset + j * CELL_SIZE
                y0 = y_offset + i * CELL_SIZE
                rect = pygame.Rect(x0, y0, CELL_SIZE, CELL_SIZE)
                if self.reached_goal:
                    pygame.draw.rect(screen, GREEN, rect, border_radius=10)
                elif (i, j) in [(0, 0), (2, 2)]:
                    pygame.draw.rect(screen, YELLOW, rect, border_radius=10)
                elif self.state[i][j] == 0:
                    pygame.draw.rect(screen, GRAY, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, LIGHT_BLUE, rect, border_radius=10)
                pygame.draw.rect(screen, DARK_GRAY, rect, 3, border_radius=10)
                if self.state[i][j] != 0:
                    text = NUMBER_FONT.render(str(self.state[i][j]), True, WHITE)
                    text_rect = text.get_rect(center=(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2))
                    screen.blit(text, text_rect)

def main():
    print("Starting main loop...")
    puzzle = Puzzle()
    running = True
    clock = pygame.time.Clock()
    try:
        while running:
            current_time = pygame.time.get_ticks()
            screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # Cập nhật trạng thái mới mỗi 2 giây nếu chưa đạt mục tiêu
            if not puzzle.reached_goal and current_time - puzzle.last_update_time >= puzzle.update_interval:
                puzzle.state = puzzle.generate_random_state()
                puzzle.last_update_time = current_time
                print(f"New random state: {puzzle.state}")
                if puzzle.is_goal():
                    puzzle.reached_goal = True
                    print("Goal state reached! Stopping updates.")

            # Vẽ tiêu đề
            title_text = TITLE_FONT.render("8-Puzzle Partially Observation", True, BLACK)
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
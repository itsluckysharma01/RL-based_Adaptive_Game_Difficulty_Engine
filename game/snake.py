import pygame
import random
import time

# Initialize pygame
pygame.init()

# Screen size
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

class SnakeGame:
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game - Baseline")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 25)
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.dx = BLOCK_SIZE
        self.dy = 0

        self.snake = [(self.x, self.y)]
        self.length = 1

        self.food = self.spawn_food()
        self.score = 0
        self.start_time = time.time()
        self.game_over = False

    def spawn_food(self):
        return (
            random.randrange(0, WIDTH, BLOCK_SIZE),
            random.randrange(0, HEIGHT, BLOCK_SIZE)
        )

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.dx == 0:
                    self.dx = -BLOCK_SIZE
                    self.dy = 0
                elif event.key == pygame.K_RIGHT and self.dx == 0:
                    self.dx = BLOCK_SIZE
                    self.dy = 0
                elif event.key == pygame.K_UP and self.dy == 0:
                    self.dx = 0
                    self.dy = -BLOCK_SIZE
                elif event.key == pygame.K_DOWN and self.dy == 0:
                    self.dx = 0
                    self.dy = BLOCK_SIZE

    def move_snake(self):
        self.x += self.dx
        self.y += self.dy

        # Wall collision
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            self.game_over = True

        # Self collision
        if (self.x, self.y) in self.snake:
            self.game_over = True

        self.snake.append((self.x, self.y))
        if len(self.snake) > self.length:
            self.snake.pop(0)

        # Food collision
        if (self.x, self.y) == self.food:
            self.length += 1
            self.score += 1
            self.food = self.spawn_food()

    def draw(self):
        self.display.fill(BLACK)

        for block in self.snake:
            pygame.draw.rect(
                self.display, GREEN,
                [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE]
            )

        pygame.draw.rect(
            self.display, RED,
            [self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE]
        )

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, (10, 10))

        pygame.display.update()

    def run(self):
        while not self.game_over:
            self.handle_input()
            self.move_snake()
            self.draw()
            self.clock.tick(10)

        pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()

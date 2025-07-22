import random
import numpy as np
import pygame
import sys

class Snake_game:
    def __init__(self, width = 400, height = 400, grid_size = 20, train_mode = True):
        pygame.init()

        self.W = width
        self.H = height
        self.GRID_SIZE = grid_size
        self.GRID_WIDTH = self.W // self.GRID_SIZE
        self.GRID_HEIGHT = self.H // self.GRID_SIZE
        self.train_mode = train_mode

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Змейка")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = (1, 0)
        self.apple = self._place_apple()
        self.score = 0
        self.steps = 0
        self.done = False
        return self._get_state()
    def _place_apple(self):
        while True:
            apple = (random.randint(0, self.GRID_WIDTH - 1),
                     random.randint(0, self.GRID_HEIGHT - 1))
            if apple not in self.snake:
                return apple
    def _get_state(self):
        state_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))

        for segment in self.snake:
            x, y = segment
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                state_grid[x, y] = 1

        apple_x, apple_y = self.apple
        state_grid[apple_x, apple_y] = 2

        state_grid[0, :] = -1
        state_grid[-1, :] = -1
        state_grid[:, 0] = -1
        state_grid[:, -1] = -1

        return state_grid

    def step(self, action=None):
        if action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != (0, 1):
                        self.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                        self.direction = (0, 1)
                    elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                        self.direction = (1, 0)
        else:
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            idx = directions.index(self.direction)
            new_dir = directions[(idx + action - 1) % 4]
            self.direction = new_dir

        head_x, head_y = self.snake[0]
        new_head = (
            head_x + self.direction[0],
            head_y + self.direction[1]
        )
        if (new_head in self.snake[1:] or
            new_head[0] < 0 or new_head[0] >= self.GRID_WIDTH
            or new_head[1] < 0 or new_head[1] >= self.GRID_HEIGHT):
            self.done = True
            return self._get_state(), -10, True

        if new_head in self.snake:
            self.done = True
            return self._get_state(), -10, self.done

        self.snake.insert(0, new_head)

        if new_head == self.apple:
            self.apple = self._place_apple()
            self.score += 1
            reward = 10
        else:
            self.snake.pop()
            reward = 0

        return self._get_state(), reward, self.done

    def step_ai(self, action):
        if self.done:
                return self._get_state(), 0, True
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = directions.index(self.direction)
        new_dir = directions[(idx + action - 1) % 4]
        self.direction = new_dir
        head_x, head_y = self.snake[0]
        new_head = (head_x + new_dir[0], head_y + new_dir[1])
        if (new_head in self.snake[1:] or
            new_head[0] < 0 or new_head[0] >= self.GRID_WIDTH
            or new_head[1] < 0 or new_head[1] >= self.GRID_HEIGHT):
            self.done = True
            return self._get_state(), -10, True
        self.snake.insert(0, new_head)
        if new_head == self.apple:
            self.apple = self._place_apple()
            self.score += 1
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
        self.steps += 1
        return self._get_state(), reward, self.done
    def render(self):
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                            (segment[0] * self.GRID_SIZE, segment[1] * self.GRID_SIZE,
                             self.GRID_SIZE, self.GRID_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (self.apple[0] * self.GRID_SIZE, self.apple[1] * self.GRID_SIZE,
                          self.GRID_SIZE, self.GRID_SIZE))
        pygame.display.flip()
        self.clock.tick(60)
    def run(self):
        while True:
            self.step()
            self.render()
            if self.done:
                print("Игра окончена! Очки:", self.score)
                self.reset()

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


pygame.init()
font = pygame.font.Font("RetroGaming.ttf", 25)


# Configure settings
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
RED = (200,0,0)
COLOR_SNAKE = (44, 195, 53)
BG_COLOR = (36, 39, 35)

BLOCK_SIZE = 20
SPEED = 40



class SnakeGameAI:
    """Set up basic Snake game"""
    def __init__(self, WIDTH=640, HEIGHT=480):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AI Snake")
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # Initialize game state
        self.direction = Direction.RIGHT

        self.head = Point(self.WIDTH/2, self.HEIGHT/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        # Randomize food placement
        x = random.randint(0, (self.WIDTH-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.HEIGHT-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        """ Implementing game rules and reward values for AI. """
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check for game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score


        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food() # Generate new food
        else:
            self.snake.pop() # Move the snake
        

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score


    def collision(self, pt=None):
        # Check for collision
        if pt is None:
            pt = self.head
        if pt.x > self.WIDTH - BLOCK_SIZE or pt.x < 0 or pt.y > self.HEIGHT - BLOCK_SIZE or pt.y < 0:
            return True # Snake hits wall
        if pt in self.snake[1:]:
            return True # Snake hits itself

        return False


    def _update_ui(self):
        self.display.fill(BG_COLOR)

        for pt in self.snake:
            pygame.draw.rect(self.display, COLOR_SNAKE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Continue straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
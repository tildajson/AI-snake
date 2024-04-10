import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Configure values
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning rate



class Agent:
    """ Create AI agent and configure settings. """
    def __init__(self):
        """ Initialize values for neural network. """
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3) # Size of input layer, hidden layer and output layer in NN
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """ Calculate game state. """
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)
        
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_right and game.collision(point_right)) or 
            (dir_left and game.collision(point_left)) or 
            (dir_up and game.collision(point_up)) or 
            (dir_down and game.collision(point_down)),

            # Danger right
            (dir_up and game.collision(point_right)) or 
            (dir_down and game.collision(point_left)) or 
            (dir_left and game.collision(point_up)) or 
            (dir_right and game.collision(point_down)),

            # Danger left
            (dir_down and game.collision(point_right)) or 
            (dir_up and game.collision(point_left)) or 
            (dir_right and game.collision(point_up)) or 
            (dir_left and game.collision(point_down)),
            
            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Location of food relative to snake head
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
            ]

        return np.array(state, dtype=int)


    def build_memory(self, state, action, reward, next_state, done):
        """
        Build memory for the agent to remember past games (long term memory), and single games (short term memory).
        """

        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            small_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            small_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*small_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        """
        Ensure the agent makes decision on what it has learned, rather than making random moves. 
        """

        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """
    Train the agent, keep track of its state and plot data into the graph.
    """

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        old_state = agent.get_state(game)
        final_move = agent.get_action(old_state)

        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        agent.build_memory(old_state, final_move, reward, new_state, done)

        if done:
            # Build long memory and plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()

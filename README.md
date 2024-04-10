# AI Snake

A version of the classic Snake game, played by AI using Reinforcement Learning (Q-Learning). The AI is a feed-forward Neural Network implemented in PyTorch.

The reward system gives +10 point for each apple eaten, and -10 point for every Game Over.

After 150 games, the AI had a highscore of 78 points and mean score of 14.9. 

## Tech Stack

+ Python
+ PyTorch
+ Matplotlib
+ Numpy

## Screenshots

[Timelapse of Training Process](https://youtu.be/GiAmK5NALZU)

## Settings

| Parameter        |      Value      |
|------------------|:---------------:|
| Hidden units number    |       256       |
| Epsilon          |   0  |
| Gamma            |       0.9       |
| Batch size       |        1000       |
| Learning rate    |       1e-3      |

## Installation

Clone this repository.

#### Install dependencies:

```bash

pip install -r requirements.txt

```

#### Run the game and start training the AI:

```bash

python main.py

```

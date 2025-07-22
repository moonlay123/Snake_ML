import snake
from agent import train_dqn
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            game = snake.Snake_game()
            game.run()
        else:
            train_dqn()
    else:
        train_dqn()
